import argparse

def init_args():
    # settings

    parser = argparse.ArgumentParser(description="Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading")
    parser.add_argument(
        "--train_start_date",
        type=str,
        default='2009-01-01',
        metavar="N",
        help="train_start_date for the model (default: '2009-01-01')",
    )
    parser.add_argument(
        "--train_end_date",
        type=str,
        default='2020-07-01',
        metavar="N",
        help="train_end_date for the model (default: '2020-07-01')",
    )
    parser.add_argument(
        "--trade_start_date",
        type=str,
        default='2020-07-01',
        metavar="N",
        help="trade_start_date for the model (default: '2020-07-01')",
    )
    parser.add_argument(
        "--trade_end_date",
        type=str,
        default='2021-10-31',
        metavar="N",
        help="trade_end_date for the model (default: '2021-10-31')",
    )
    parser.add_argument(
        "--total_timesteps",
        type=float,
        default=40000,
        metavar="N",
        help="total_timesteps for the training (default: 40000)",
    )
    args = parser.parse_args()

    return args