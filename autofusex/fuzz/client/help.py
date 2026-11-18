"""Scenario-oriented help, keeping the actual command paths stable."""
import argparse


class ScenarioParser(argparse.ArgumentParser):
    def format_help(self):
        return '''usage: afx fuzz COMMAND [OPTIONS]

客户端 · 执行、采集、保存
  run [SEED ...] [--count N]               后台测试；不传 --count 无限运行，--save 才保存
  replay SEED [--save]            单 seed 重放
  status                                 查看任务状态
  stop                                   停止当前测试

云端上报 · 见 afx platform client -h（上传、心跳、复测）

本地查看 · 无需 Torch / CANN / NPU / Node
  view RESULTS.sqlite                    浏览器查看可信 SQLite 快照
  maintenance snapshot SOURCE DEST       导出可分享的轻量结果快照

维护 · 诊断、索引、恢复
  maintenance -h                         列出全部维护命令

使用 afx fuzz COMMAND -h 查看参数。
完整说明：docs/fuzz/fuzz_usage.md
'''
