import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description="Kernel Estimation Parameters")
    parser.add_argument('--p', type=int, default=25, help='Kernel size parameter')
    # not finished yet
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    print(f"Kernel size parameter p: {args.p}")