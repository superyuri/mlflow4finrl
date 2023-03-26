import argparse

def init_args():
    # settings

    parser = argparse.ArgumentParser(description="Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading")
    parser.add_argument(
        "--train_start_date",
        type=str,
        default='2022-07-01',
        metavar="N",
        help="train_start_date for the model (default: '2022-07-01')",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        default='2022-08-31',
        metavar="N",
        help="train_end_date for the model (default: '2022-08-31')",
    )
    parser.add_argument(
        "--test_start_date",
        type=str,
        default='2022-09-01',
        metavar="N",
        help="test_start_date for the model (default: '2022-09-01')",
    )
    parser.add_argument(
        "--test_end_date",
        type=str,
        default='2022-09-30',
        metavar="N",
        help="test_end_date for the model (default: '2022-09-30')",
    )
    parser.add_argument(
        "--total_timesteps",
        type=float,
        default=1000,
        metavar="N",
        help="total_timesteps for the training (default: 1000)",
    )
    args = parser.parse_args()

    return args