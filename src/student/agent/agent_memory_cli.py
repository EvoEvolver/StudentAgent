#!/usr/bin/env python3
"""
CLI for AgentMemoryV2 system.
Allows users to learn, retrieve, and list memories from the command line.
"""

import argparse
import sys
from pathlib import Path
from agent_memory_v2 import AgentMemoryV2


def remember(memory: AgentMemoryV2, content: str):
    """Learn and store new content."""
    print("Learning new information...")
    title = memory.learn(content)
    print(f"✓ Stored with title: '{title}'")


def retrieve(memory: AgentMemoryV2, query: str, top_k: int = None):
    """Retrieve memories based on a query."""
    print(f"Searching for: '{query}'")
    results = memory.retrieve(query, top_k=top_k)

    if not results:
        print("No relevant memories found.")
        return

    print(f"\nFound {len(results)} relevant memor{'y' if len(results) == 1 else 'ies'}:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Title: {result['title']}")
        print(f"   Content: {result['content']}")
        print()


def list_all(memory: AgentMemoryV2):
    """List all stored memory titles."""
    memories = memory.get_all_memories()

    if not memories:
        print("No memories stored yet.")
        return

    print(f"Stored memories ({len(memories)} total):\n")
    for i, (title, content) in enumerate(memories.items(), 1):
        print(f"{i}. {title}")
        preview = content[:80] + "..." if len(content) > 80 else content
        print(f"   Preview: {preview}")
        print()


def delete(memory: AgentMemoryV2, title: str):
    """Delete a specific memory by title."""
    if memory.delete_memory(title):
        print(f"✓ Deleted memory: '{title}'")
    else:
        print(f"✗ Memory not found: '{title}'")


def clear_all(memory: AgentMemoryV2):
    """Clear all memories after confirmation."""
    confirm = input("Are you sure you want to clear ALL memories? (yes/no): ")
    if confirm.lower() in ['yes', 'y']:
        memory.clear()
        print("✓ All memories cleared.")
    else:
        print("Cancelled.")


def interactive_mode(memory: AgentMemoryV2):
    """Run interactive CLI mode."""
    print("=== Agent Memory System ===")
    print("Interactive Mode\n")

    while True:
        print("\nCommands:")
        print("  1. Remember new information")
        print("  2. Retrieve memories")
        print("  3. List all memories")
        print("  4. Delete a memory")
        print("  5. Clear all memories")
        print("  6. Exit")

        choice = input("\nEnter command (1-6): ").strip()

        if choice == "1":
            content = input("\nEnter content to remember: ").strip()
            if content:
                remember(memory, content)
            else:
                print("Content cannot be empty.")

        elif choice == "2":
            query = input("\nEnter search query: ").strip()
            if query:
                try:
                    top_k_input = input("Max results (press Enter for all): ").strip()
                    top_k = int(top_k_input) if top_k_input else None
                    retrieve(memory, query, top_k)
                except ValueError:
                    print("Invalid number, showing all results.")
                    retrieve(memory, query)
            else:
                print("Query cannot be empty.")

        elif choice == "3":
            list_all(memory)

        elif choice == "4":
            list_all(memory)
            title = input("\nEnter title to delete: ").strip()
            if title:
                delete(memory, title)
            else:
                print("Title cannot be empty.")

        elif choice == "5":
            clear_all(memory)

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid command. Please enter 1-6.")


def main():
    parser = argparse.ArgumentParser(
        description="Agent Memory System CLI - Store and retrieve information using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python agent_memory_cli.py

  # Remember new information
  python agent_memory_cli.py remember "Paris is the capital of France"

  # Retrieve memories
  python agent_memory_cli.py retrieve "European capitals"

  # List all memories
  python agent_memory_cli.py list

  # Delete a memory
  python agent_memory_cli.py delete "Title of memory"

  # Clear all memories
  python agent_memory_cli.py clear
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["remember", "retrieve", "list", "delete", "clear"],
        help="Command to execute (omit for interactive mode)"
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Content to remember or query to search (depending on command)"
    )
    parser.add_argument(
        "--storage",
        default="agent_memory.json",
        help="Path to storage file (default: agent_memory.json)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Maximum number of results to retrieve"
    )

    args = parser.parse_args()

    # Initialize memory system
    memory = AgentMemoryV2(
        model=args.model,
        storage_file=args.storage
    )

    # Execute command or enter interactive mode
    if not args.command:
        interactive_mode(memory)
    elif args.command == "remember":
        if not args.text:
            print("Error: Please provide content to remember")
            sys.exit(1)
        content = " ".join(args.text)
        remember(memory, content)
    elif args.command == "retrieve":
        if not args.text:
            print("Error: Please provide a search query")
            sys.exit(1)
        query = " ".join(args.text)
        retrieve(memory, query, args.top_k)
    elif args.command == "list":
        list_all(memory)
    elif args.command == "delete":
        if not args.text:
            print("Error: Please provide a title to delete")
            sys.exit(1)
        title = " ".join(args.text)
        delete(memory, title)
    elif args.command == "clear":
        clear_all(memory)


if __name__ == "__main__":
    main()
