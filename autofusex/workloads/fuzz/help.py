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
  resume                                 恢复中断任务
  seed TOKEN [--show-spec]                查看固定 seed 规格
  diagnose EXPECTED ACTUAL               比较生成轨迹

云端上报 · 见 afx sync -h 和 afx retest -h

本地查看 · 无需 Torch / CANN / NPU / Node
  afx view RESULTS.sqlite                    浏览器查看可信 SQLite 快照
  afx results snapshot SOURCE DEST       导出可分享的轻量结果快照

维护 · 诊断、索引、恢复
  afx results -h                         列出结果维护命令
  afx doctor --workload fuzz              检查环境和生成器版本

使用 afx fuzz COMMAND -h 查看参数。
完整说明：docs/fuzz/fuzz_usage.md
'''
