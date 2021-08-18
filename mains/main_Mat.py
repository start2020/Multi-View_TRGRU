import libs.main_common, libs.para
import argparse

def main(args):
    libs.main_common.regression_main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = libs.para.Mat_main(parser)
    parser = libs.para.common_para(parser)
    args = parser.parse_args()
    main(args)