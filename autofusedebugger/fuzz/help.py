"""Scenario-oriented help, keeping the actual command paths stable."""
import argparse


class ScenarioParser(argparse.ArgumentParser):
    def format_help(self):
        return '''usage: autofusedbg fuzz COMMAND [OPTIONS]

客户端 · 执行、采集、自动上报
  setup --token TOKEN [--sync-interval 30]  初始化；用户名由云端解析
  run [SEED ...] [--count N]               后台测试；不传 --count 无限运行，--save 才保存/上传
  replay SEED [--save]            单 seed 重放
  status                                 查看任务状态
  stop                                   停止当前测试

云端 · 用户、汇总、网站
  admin user DATABASE NAME               创建用户名，输出一次 token
  admin rotate-token DATABASE NAME       轮换 token
  admin serve DATABASE [--port 8090]      接收上报并提供网页，需 TLS 代理
  admin backup DATABASE DESTINATION      私有一致性备份，含身份及 ACK

本地查看 · 无需 Torch / CANN / NPU / Node
  view RESULTS.sqlite                    浏览器查看可信 SQLite 快照
  maintenance snapshot SOURCE DEST       导出可分享的轻量结果快照

维护 · 诊断、同步、恢复
  maintenance -h                         列出全部维护命令

使用 autofusedbg fuzz COMMAND -h 查看参数；云端命令用 admin COMMAND -h。
完整说明：docs/af_fuzz_usage.md
'''
