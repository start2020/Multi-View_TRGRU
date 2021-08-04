from libs import main_common, para
import argparse

def main(args):
    main_common.regression_main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = para.Lasso_main(parser)
    parser = para.common_para(parser)
    args = parser.parse_args()
    main(args)
