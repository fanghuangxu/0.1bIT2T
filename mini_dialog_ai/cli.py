import argparse
import sys
from .model import DialogAI

def main():
    parser = argparse.ArgumentParser(description="Mini Dialog AI - A lightweight conversational AI")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu or cuda")
    args = parser.parse_args()

    print("Initializing Dialog AI...")
    ai = DialogAI(device=args.device)
    print("Dialog AI initialized!\n")
    
    while True:
        print("=" * 50)
        print("1. Text to Text")
        print("2. Image + Text to Text")
        print("3. Summarize Conversation")
        print("4. Clear History")
        print("5. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            message = input("Enter your message: ")
            response = ai.chat(message)
            print(f"\nAI: {response}\n")
        
        elif choice == "2":
            image_path = input("Enter image path: ")
            message = input("Enter your question (optional): ")
            response = ai.chat(message, image_path)
            print(f"\nAI: {response}\n")
        
        elif choice == "3":
            summary = ai.summarize_conversation()
            print(f"\nConversation Summary:\n{summary}\n")
        
        elif choice == "4":
            ai.clear_history()
            print("\nConversation history cleared!\n")
        
        elif choice == "5":
            print("Goodbye!")
            sys.exit(0)
        
        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    main()