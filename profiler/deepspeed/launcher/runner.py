import sys
from hook import inject_hook, spawn_process

def main():
    inject_hook()

    spawn_process()


if __name__ == "__main__":
    main()