import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from agency_swarm import Agency, Agent
from agency_swarm.thread import ConversationThread

# --- File Persistence Setup ---


@pytest.fixture(scope="function")
def temp_persistence_dir(tmp_path):
    print(f"\nTEMP DIR: Created at {tmp_path}")
    yield tmp_path


def file_save_callback(thread: ConversationThread, base_dir: Path):
    if not isinstance(thread, ConversationThread):
        print(f"FILE SAVE ERROR: Expected ConversationThread, got {type(thread)}")
        return
    thread_id = thread.thread_id
    if not thread_id:
        print("FILE SAVE ERROR: Thread has no ID. Cannot save.")
        return

    file_path = base_dir / f"{thread_id}.json"
    print(f"\nFILE SAVE: Saving thread '{thread_id}' to {file_path}")
    try:
        thread_dict = {
            "thread_id": thread.thread_id,
            "items": [item for item in thread.items],
            "metadata": thread.metadata,
        }
        with open(file_path, "w") as f:
            json.dump(thread_dict, f, indent=2)
        print(f"FILE SAVE: Successfully saved {file_path}")
    except Exception as e:
        print(f"FILE SAVE ERROR: Failed to save thread {thread_id}: {e}")
        import traceback

        traceback.print_exc()


def file_load_callback(thread_id: str, base_dir: Path) -> Optional[ConversationThread]:
    file_path = base_dir / f"{thread_id}.json"
    print(f"\nFILE LOAD: Attempting to load thread '{thread_id}' from {file_path}")
    if not file_path.exists():
        print("FILE LOAD: File not found.")
        return None
    try:
        with open(file_path) as f:
            thread_dict = json.load(f)
        loaded_thread = ConversationThread(
            thread_id=thread_dict.get("thread_id", thread_id),
            items=thread_dict.get("items", []),
            metadata=thread_dict.get("metadata", {}),
        )
        print(f"FILE LOAD: Successfully loaded and reconstructed thread '{thread_id}'")
        return loaded_thread
    except Exception as e:
        print(f"FILE LOAD ERROR: Failed to load/reconstruct thread {thread_id}: {e}")
        # Log traceback for detailed debugging
        import traceback

        traceback.print_exc()
        return None


def file_save_callback_error(thread: ConversationThread, base_dir: Path):
    """Mock file save callback that raises an error."""
    thread_id = thread.thread_id
    if not thread_id:
        print("FILE SAVE ERROR (Intentional Fail): Thread has no ID.")
        raise ValueError("Cannot simulate save error for thread without ID")

    file_path = base_dir / f"{thread_id}.json"
    print(f"\nFILE SAVE ERROR: Intentionally failing for thread '{thread_id}' at {file_path}")
    raise IOError(f"Simulated save error for {thread_id}")


def file_load_callback_error(thread_id: str, base_dir: Path) -> Optional[ConversationThread]:
    """Mock file load callback that raises an error."""
    file_path = base_dir / f"{thread_id}.json"
    print(f"\nFILE LOAD ERROR: Intentionally failing for thread '{thread_id}' at {file_path}")
    raise IOError(f"Simulated load error for {thread_id}")


# --- Test Agent ---
class PersistenceTestAgent(Agent):
    pass


@pytest.fixture
def persistence_agent():
    return PersistenceTestAgent(
        name="PersistenceTester",
        instructions="Remember the secret code word I tell you. In the next turn, repeat the code word.",
    )


@pytest.fixture
def file_persistence_callbacks(temp_persistence_dir):
    """Fixture to provide configured file callbacks."""
    save_cb = lambda thread: file_save_callback(thread, temp_persistence_dir)
    load_cb = lambda thread_id: file_load_callback(thread_id, temp_persistence_dir)
    return load_cb, save_cb


# --- Removed InputCaptureHook ---

# --- Test Cases ---


@pytest.mark.asyncio
async def test_persistence_callbacks_called(temp_persistence_dir, persistence_agent):
    """
    Test that save and load callbacks are invoked, checking side effects (file existence).
    """
    chat_id = "callback_test_1"
    message1 = "First message for callback test."
    message2 = "Second message for callback test."
    chat_file = temp_persistence_dir / f"{chat_id}.json"

    # Create actual callbacks first
    actual_save_cb = lambda thread: file_save_callback(thread, base_dir=temp_persistence_dir)
    actual_load_cb = lambda thread_id: file_load_callback(thread_id, base_dir=temp_persistence_dir)

    # Patch the target functions *before* Agency instantiation
    with (
        patch("tests.integration.test_persistence.file_load_callback") as load_spy,
        patch("tests.integration.test_persistence.file_save_callback") as save_spy,
    ):
        # Configure spies to call the actual implementations when they are called
        load_spy.side_effect = actual_load_cb
        save_spy.side_effect = actual_save_cb

        # Pass the *spies* themselves as the callbacks to Agency
        agency = Agency(
            agency_chart=[persistence_agent],
            load_callback=load_spy,  # Pass the spy
            save_callback=save_spy,  # Pass the spy
        )

        # Turn 1
        print(f"\n--- Callback Test Turn 1 (ChatID: {chat_id}) --- MSG: {message1}")
        assert not chat_file.exists(), f"File {chat_file} should not exist before first run."
        await agency.get_response(
            message=message1, recipient_agent=persistence_agent.name, chat_id=chat_id
        )

        # Verify load was attempted (called by get_thread)
        load_spy.assert_called()
        # Verify save was attempted (called by get_thread on create AND by PersistenceHooks on end)
        save_spy.assert_called()
        assert save_spy.call_count >= 1  # Should be called at least once (might be twice)
        # Verify save *succeeded* by checking file existence
        assert chat_file.exists(), f"File {chat_file} should exist after first run."

        # Read file content to check basic structure (optional)
        with open(chat_file) as f:
            data = json.load(f)
        assert data.get("thread_id") == chat_id
        assert isinstance(data.get("items"), list)

        # Turn 2
        print(f"\n--- Callback Test Turn 2 (ChatID: {chat_id}) --- MSG: {message2}")
        # Use a new agency instance to ensure loading happens via the callback
        # Re-patch for the new instance
        with (
            patch("tests.integration.test_persistence.file_load_callback") as load_spy_2,
            patch("tests.integration.test_persistence.file_save_callback") as save_spy_2,
        ):
            load_spy_2.side_effect = actual_load_cb
            save_spy_2.side_effect = actual_save_cb

            agency2 = Agency(
                agency_chart=[persistence_agent], load_callback=load_spy_2, save_callback=save_spy_2
            )
            await agency2.get_response(
                message=message2, recipient_agent=persistence_agent.name, chat_id=chat_id
            )

            # Verify load was called again for turn 2
            load_spy_2.assert_called()
            # Verify save was called again for turn 2
            save_spy_2.assert_called()
            assert save_spy_2.call_count >= 1  # At least once for end hook, maybe more
            # Verify file still exists (implicitly means save likely worked again)
            assert chat_file.exists(), f"File {chat_file} should still exist after second run."


@pytest.mark.asyncio
async def test_persistence_loads_history(file_persistence_callbacks, persistence_agent):
    """
    Test that history is correctly loaded from file and present in the thread
    before the next turn starts.
    """
    load_cb, save_cb = file_persistence_callbacks
    chat_id = "history_test_1"
    secret_code = "platypus7"
    message1_content = f"The secret code word is {secret_code}. Remember it."
    message2_content = "What was the secret code word?"

    # Agency Instance 1 - Turn 1
    print("\n--- History Test Instance 1 - Turn 1 --- Creating Agency 1")
    agency1 = Agency(agency_chart=[persistence_agent], load_callback=load_cb, save_callback=save_cb)
    print(f"--- History Test Instance 1 - Turn 1 (ChatID: {chat_id}) --- MSG: {message1_content}")
    await agency1.get_response(
        message=message1_content, recipient_agent="PersistenceTester", chat_id=chat_id
    )

    # --- Verification after Turn 1 ---
    # Manually load the thread data using the callback to check saved state
    print(f"\n--- Verifying saved state for {chat_id} after Turn 1 ---")
    loaded_thread_after_t1 = load_cb(chat_id)  # Now returns ConversationThread or None
    assert loaded_thread_after_t1 is not None, "Thread data failed to load after Turn 1"
    assert isinstance(
        loaded_thread_after_t1, ConversationThread
    ), f"Loaded data is not a ConversationThread: {type(loaded_thread_after_t1)}"
    # Check if message 1 is in the saved items
    found_message1_in_saved = False
    items_to_check = loaded_thread_after_t1.items
    # Print items for debugging
    print(f"DEBUG: Saved items for {chat_id}: {items_to_check}")
    for item_dict in items_to_check:
        if (
            isinstance(item_dict, dict)
            and item_dict.get("role") == "user"
            and message1_content in item_dict.get("content", "")
        ):
            found_message1_in_saved = True
            break
    assert (
        found_message1_in_saved
    ), f"Message 1 content '{message1_content}' not found in saved thread items: {items_to_check}"
    print("--- Saved state verified successfully. ---")

    # Agency Instance 2 - Turn 2
    print("\n--- History Test Instance 2 - Turn 2 --- Creating Agency 2")
    agency2 = Agency(agency_chart=[persistence_agent], load_callback=load_cb, save_callback=save_cb)
    print(f"--- History Test Instance 2 - Turn 2 (ChatID: {chat_id}) --- MSG: {message2_content}")
    await agency2.get_response(
        message=message2_content, recipient_agent="PersistenceTester", chat_id=chat_id
    )

    # --- Verification after Turn 2 ---
    # Access the thread directly from the second agency's manager
    print(f"\n--- Verifying loaded state in Agency 2 for {chat_id} after Turn 2 ---")
    thread_in_agency2 = agency2.thread_manager._threads.get(chat_id)
    assert thread_in_agency2 is not None, f"Thread {chat_id} not found in agency2's thread manager."
    assert isinstance(thread_in_agency2, ConversationThread)
    # Check if message 1 is STILL in the items list after loading and Turn 2 execution
    found_message1_in_loaded = False
    items_to_check_loaded = thread_in_agency2.items
    # Print items for debugging
    print(f"DEBUG: Loaded items in agency2 for {chat_id}: {items_to_check_loaded}")
    for item in items_to_check_loaded:
        if (
            isinstance(item, dict)
            and item.get("role") == "user"
            and message1_content in item.get("content", "")
        ):
            found_message1_in_loaded = True
            break
    assert found_message1_in_loaded, f"Message 1 content '{message1_content}' not found in loaded thread items in agency2: {items_to_check_loaded}"
    print("--- Loaded state verified successfully. ---")

    # Final LLM output content is NOT asserted.


@pytest.mark.asyncio
async def test_persistence_load_error(temp_persistence_dir, persistence_agent):
    """
    Test that Agency.get_response fails gracefully if the load_callback raises an error.
    """
    chat_id = "load_error_test_1"
    message1 = "Message before load error."

    # Setup faulty callbacks
    error_load_cb = lambda tid: file_load_callback_error(tid, temp_persistence_dir)
    # Use normal save for the first turn to create a file to load
    normal_save_cb = lambda t: file_save_callback(t, temp_persistence_dir)

    # Agency Instance 1 - Turn 1 (Normal save)
    print("\n--- Load Error Test Instance 1 - Turn 1 --- Creating Agency 1")
    agency1 = Agency(
        agency_chart=[persistence_agent],
        load_callback=None,  # No load on first run
        save_callback=normal_save_cb,
    )
    print(f"--- Load Error Test Instance 1 - Turn 1 (ChatID: {chat_id}) --- MSG: {message1}")
    await agency1.get_response(
        message=message1, recipient_agent="PersistenceTester", chat_id=chat_id
    )
    print("--- Load Error Test Instance 1 - Turn 1 Completed ---")

    # Agency Instance 2 - Turn 2 (Error on load)
    print("\n--- Load Error Test Instance 2 - Turn 2 --- Creating Agency 2 (with error load)")
    # Patch load callback *before* agency init
    with patch("tests.integration.test_persistence.file_load_callback_error") as error_load_spy:
        error_load_spy.side_effect = error_load_cb
        agency2 = Agency(
            agency_chart=[persistence_agent],
            load_callback=error_load_spy,  # Pass the spy
            save_callback=normal_save_cb,  # Save doesn't matter here
        )
        print(f"--- Load Error Test Instance 2 - Turn 2 (ChatID: {chat_id}) --- Expecting Error")
        with pytest.raises(IOError, match=f"Simulated load error for {chat_id}"):
            await agency2.get_response(
                message="This message should not be processed.",
                recipient_agent="PersistenceTester",
                chat_id=chat_id,
            )
    print("--- Load Error Test Instance 2 - Caught expected IOError ---")


@pytest.mark.asyncio
async def test_persistence_save_error(temp_persistence_dir, persistence_agent):
    """
    Test that Agency.get_response completes but logs an error if the save_callback fails.
    Expect logger to be called TWICE now (once on create, once on end hook).
    """
    chat_id = "save_error_test_1"
    message1 = "Message causing save error."

    # Setup faulty save callback
    error_save_cb = lambda t: file_save_callback_error(t, temp_persistence_dir)
    # Load doesn't matter here
    normal_load_cb = lambda tid: file_load_callback(tid, temp_persistence_dir)

    # Agency Instance
    print("\n--- Save Error Test Instance - Turn 1 --- Creating Agency")
    agency = Agency(
        agency_chart=[persistence_agent],
        load_callback=normal_load_cb,
        save_callback=error_save_cb,  # Use the error callback here
    )

    print(f"--- Save Error Test Instance - Turn 1 (ChatID: {chat_id}) --- Expecting Save Error Log")
    # Patch the correct logger
    with patch("agency_swarm.thread.logger.error") as mock_logger_error:
        # Run should complete successfully
        result = await agency.get_response(
            message=message1, recipient_agent="PersistenceTester", chat_id=chat_id
        )
        # Assert that the agent interaction produced a result (run completed)
        assert result is not None
        # Check the type is RunResult
        from agents import RunResult  # Import RunResult

        assert isinstance(result, RunResult)
        # Check the final output attribute
        assert hasattr(result, "final_output")
        assert isinstance(result.final_output, str)

        print("--- Save Error Test Instance - Turn 1 Completed Successfully (as expected) ---")

        # Verify logger.error was called TWICE due to save_callback failure
        assert (
            mock_logger_error.call_count == 2
        ), f"Expected logger.error to be called twice, but was called {mock_logger_error.call_count} times."

        # Check details of the calls (optional, check first or last call)
        # First call args (from ThreadManager save on create)
        args1, kwargs1 = mock_logger_error.call_args_list[0]
        assert "Error saving thread" in args1[0]
        assert chat_id in args1[0]
        # assert isinstance(args1[1], IOError) # Exception object not passed directly
        assert f"Simulated save error for {chat_id}" in args1[0]
        assert kwargs1.get("exc_info") is True  # Check exc_info was passed

        # Second call args (from PersistenceHooks save on end)
        args2, kwargs2 = mock_logger_error.call_args_list[1]
        assert "Error saving thread" in args2[0]
        assert chat_id in args2[0]
        # assert isinstance(args2[1], IOError) # Exception object not passed directly
        assert f"Simulated save error for {chat_id}" in str(args2[0])  # Check error in message
        assert kwargs2.get("exc_info") is True  # Check exc_info was passed

    print("--- Save Error Test Instance - Verified logger.error calls ---")

    # Additionally, check that the file wasn't actually created due to the error
    error_file_path = temp_persistence_dir / f"{chat_id}.json"
    assert (
        not error_file_path.exists()
    ), f"File {error_file_path} should not exist after save error."
    print("--- Save Error Test Instance - Verified file does not exist ---")


@pytest.mark.asyncio
async def test_persistence_chat_id_isolation(
    file_persistence_callbacks, persistence_agent, temp_persistence_dir
):
    """
    Test that persistence for one chat_id does not interfere with another.
    Uses the file-based persistence which saves one file per chat_id.
    """
    load_cb, save_cb = file_persistence_callbacks
    chat_id_1 = "isolation_test_1"
    chat_id_2 = "isolation_test_2"
    message_1a = "Message for chat 1, first turn."
    message_1b = "Message for chat 1, second turn."
    message_2a = "Message for chat 2, first turn."

    # Agency Instance
    print("\\n--- Isolation Test - Creating Agency ---")
    agency = Agency(agency_chart=[persistence_agent], load_callback=load_cb, save_callback=save_cb)

    # Turn 1 for Chat 1
    print(f"--- Isolation Test - Turn 1 (ChatID: {chat_id_1}) --- MSG: {message_1a}")
    await agency.get_response(
        message=message_1a, recipient_agent="PersistenceTester", chat_id=chat_id_1
    )
    chat1_file = temp_persistence_dir / f"{chat_id_1}.json"
    assert chat1_file.exists(), f"File for {chat_id_1} should exist after turn 1."

    # Turn 1 for Chat 2
    print(f"--- Isolation Test - Turn 1 (ChatID: {chat_id_2}) --- MSG: {message_2a}")
    await agency.get_response(
        message=message_2a, recipient_agent="PersistenceTester", chat_id=chat_id_2
    )
    chat2_file = temp_persistence_dir / f"{chat_id_2}.json"
    assert chat2_file.exists(), f"File for {chat_id_2} should exist after turn 1."
    assert chat1_file.exists(), f"File for {chat_id_1} should still exist after chat 2's turn 1."

    # Turn 2 for Chat 1
    print(f"--- Isolation Test - Turn 2 (ChatID: {chat_id_1}) --- MSG: {message_1b}")
    # Create a new agency instance to force loading
    agency_reloaded = Agency(
        agency_chart=[persistence_agent], load_callback=load_cb, save_callback=save_cb
    )
    await agency_reloaded.get_response(
        message=message_1b, recipient_agent="PersistenceTester", chat_id=chat_id_1
    )

    # Verify Chat 1's history contains message 1a, but not message 2a
    print(f"--- Isolation Test - Verifying loaded state for {chat_id_1} ---")
    loaded_thread_1 = load_cb(chat_id_1)
    assert loaded_thread_1 is not None
    assert isinstance(loaded_thread_1, ConversationThread)
    found_message_1a = any(
        item.get("role") == "user" and message_1a in item.get("content", "")
        for item in loaded_thread_1.items
    )
    found_message_2a = any(
        item.get("role") == "user" and message_2a in item.get("content", "")
        for item in loaded_thread_1.items
    )
    assert (
        found_message_1a
    ), f"Message '{message_1a}' not found in loaded thread for {chat_id_1}: {loaded_thread_1.items}"
    assert not found_message_2a, f"Message '{message_2a}' (from chat 2) unexpectedly found in loaded thread for {chat_id_1}: {loaded_thread_1.items}"
    print(f"--- Isolation Test - Verification for {chat_id_1} successful ---")

    # Verify Chat 2's history contains message 2a, but not message 1a or 1b
    print(f"--- Isolation Test - Verifying loaded state for {chat_id_2} ---")
    loaded_thread_2 = load_cb(chat_id_2)
    assert loaded_thread_2 is not None
    assert isinstance(loaded_thread_2, ConversationThread)
    items_2 = loaded_thread_2.items
    found_message_1a_in_2 = any(
        item.get("role") == "user" and message_1a in item.get("content", "") for item in items_2
    )
    found_message_1b_in_2 = any(
        item.get("role") == "user" and message_1b in item.get("content", "") for item in items_2
    )
    found_message_2a_in_2 = any(
        item.get("role") == "user" and message_2a in item.get("content", "") for item in items_2
    )
    assert (
        found_message_2a_in_2
    ), f"Message '{message_2a}' not found in loaded thread for {chat_id_2}: {items_2}"
    assert not found_message_1a_in_2, f"Message '{message_1a}' (from chat 1) unexpectedly found in loaded thread for {chat_id_2}: {items_2}"
    assert not found_message_1b_in_2, f"Message '{message_1b}' (from chat 1) unexpectedly found in loaded thread for {chat_id_2}: {items_2}"
    print(f"--- Isolation Test - Verification for {chat_id_2} successful ---")


@pytest.mark.asyncio
async def test_no_persistence_no_callbacks(persistence_agent, temp_persistence_dir):
    """
    Test that history is NOT persisted between Agency instances if no callbacks are provided.
    """
    chat_id = "no_persistence_test_1"
    message1 = "First message, should be forgotten."
    message2 = "Second message, load should not happen."

    # Agency Instance 1 - Turn 1 (No callbacks)
    print("\\n--- No Persistence Test - Instance 1 - Turn 1 --- Creating Agency 1")
    agency1 = Agency(agency_chart=[persistence_agent], load_callback=None, save_callback=None)
    print(f"--- No Persistence Test - Instance 1 - Turn 1 (ChatID: {chat_id}) --- MSG: {message1}")
    await agency1.get_response(
        message=message1, recipient_agent="PersistenceTester", chat_id=chat_id
    )

    # Check that no file was created (as no save callback was provided)
    persistence_file = temp_persistence_dir / f"{chat_id}.json"
    assert not persistence_file.exists(), f"Persistence file {persistence_file} should NOT exist."
    print("--- No Persistence Test - Verified no file created after Turn 1 ---")

    # Agency Instance 2 - Turn 2 (No callbacks)
    print("\\n--- No Persistence Test - Instance 2 - Turn 2 --- Creating Agency 2")
    agency2 = Agency(agency_chart=[persistence_agent], load_callback=None, save_callback=None)
    print(f"--- No Persistence Test - Instance 2 - Turn 2 (ChatID: {chat_id}) --- MSG: {message2}")
    await agency2.get_response(
        message=message2, recipient_agent="PersistenceTester", chat_id=chat_id
    )

    # Verify the thread in agency2 only contains message2, not message1
    thread_in_agency2 = agency2.thread_manager._threads.get(chat_id)
    assert thread_in_agency2 is not None
    found_message1 = any(
        item.get("role") == "user" and message1 in item.get("content", "")
        for item in thread_in_agency2.items
    )
    found_message2 = any(
        item.get("role") == "user" and message2 in item.get("content", "")
        for item in thread_in_agency2.items
    )

    assert (
        not found_message1
    ), f"Message '{message1}' (from instance 1) was unexpectedly found in instance 2."
    assert found_message2, f"Message '{message2}' not found in instance 2 thread."
    print("--- No Persistence Test - Verified thread history in instance 2 ---")
