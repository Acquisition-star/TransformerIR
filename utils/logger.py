import os
import sys
import logging
import functools
from colorama import Fore, Style


class MultiColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)

    def format(self, record):
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        # 时间、日志名称 青色
        asctime = f"{Fore.CYAN}[{record.asctime} {record.name}]{Style.RESET_ALL}"
        # 文件名和行号颜色
        filename_lineno = f"{Fore.CYAN}({record.filename} {record.lineno}){Style.RESET_ALL}"
        # 设置 INFO 颜色
        levelname = f"{Fore.GREEN}{record.levelname}{Style.RESET_ALL}"
        # 消息内容
        message = record.getMessage()

        # 拼接格式化的日志字符串
        log_msg = f"{asctime} {filename_lineno}: {levelname} {message}"
        return log_msg


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    """
    创建日志对象
    :param output_dir: 日志文件保存路径
    :param dist_rank: 当前进程
    :param name: 日志名称
    :return: 配置好的日志对象
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置日志器级别
    logger.propagate = False  # 禁止日志器将日志传播至父日志

    # 日志格式
    format = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'  # [时间 命名] (文件名 行号): 日志级别 日志内容
    datefmt = '%Y-%m-%d %H:%M:%S'

    # 创建终端日志输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # 设置终端日志级别为 DEBUG
    console_handler.setFormatter(MultiColoredFormatter(fmt=format, datefmt=datefmt))
    logger.addHandler(console_handler)

    # 创建文件日志输出
    file_handler = logging.FileHandler(os.path.join(output_dir, f'record.log'), mode='a')
    file_handler.setLevel(logging.DEBUG)  # 设置文件日志级别为 DEBUG
    file_formatter = logging.Formatter(fmt=format, datefmt=datefmt)
    file_handler.setFormatter(file_formatter)  # 使用普通格式器
    logger.addHandler(file_handler)

    return logger
