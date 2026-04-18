#!/usr/bin/env python3
"""Telegram bot to monitor Oracle ARM retry scripts on remote VMs."""

import logging
import paramiko
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = "5939673882:AAF8v9NIcP-RP_rOOlPiDVU0lJx0QbowbWc"
SSH_KEY_PATH = "/home/ubuntu/.ssh/monitor_key"

VMS = [
    {"name": "Frankfurt", "host": "130.162.63.177", "user": "ubuntu"},
    {"name": "Ashburn",   "host": "150.136.246.222", "user": "ubuntu"},
    {"name": "London",    "host": "150.230.123.103", "user": "ubuntu"},
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_vm(vm: dict) -> dict:
    """SSH into a VM and gather retry script status."""
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            vm["host"], username=vm["user"],
            key_filename=SSH_KEY_PATH, timeout=10,
        )

        def run(cmd):
            _, stdout, _ = client.exec_command(cmd, timeout=10)
            return stdout.read().decode().strip()

        # Check if script is running
        is_running = run("pgrep -f retry_a1 >/dev/null 2>&1 && echo YES || echo NO") == "YES"

        # Get current attempt number from last log line
        last_attempt_line = run("grep 'Attempt #' ~/retry.log 2>/dev/null | tail -1")
        attempt_num = "0"
        if "Attempt #" in last_attempt_line:
            try:
                attempt_num = last_attempt_line.split("Attempt #")[1].split(" ")[0].strip()
            except (IndexError, ValueError):
                attempt_num = "?"

        # Total log lines (each attempt = 3 ADs logged)
        total_log_lines = run("wc -l < ~/retry.log 2>/dev/null || echo 0")

        # Last 5 lines of log
        last_lines = run("tail -5 ~/retry.log 2>/dev/null || echo 'No log'")

        # Check for SUCCESS
        success = run("test -f ~/SUCCESS.txt && cat ~/SUCCESS.txt || echo ''")

        client.close()

        return {
            "name": vm["name"],
            "host": vm["host"],
            "running": is_running,
            "attempt": attempt_num,
            "log_lines": total_log_lines,
            "last_lines": last_lines,
            "success": success,
        }
    except Exception as e:
        return {
            "name": vm["name"],
            "host": vm["host"],
            "running": False,
            "error": str(e),
        }


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command — check all VMs."""
    msg = await update.message.reply_text("⏳ Checking VMs...")

    lines = ["🖥 *Oracle ARM Retry Status*\n"]
    for vm in VMS:
        r = check_vm(vm)
        if "error" in r:
            lines.append(f"❌ *{r['name']}* (`{r['host']}`)")
            lines.append(f"   Error: `{r['error'][:80]}`\n")
        elif r["success"]:
            lines.append(f"🎉 *{r['name']}* — *SUCCESS!*")
            lines.append(f"   `{r['success'][:120]}`\n")
        else:
            icon = "✅" if r["running"] else "🔴"
            status_text = "Running" if r["running"] else "STOPPED"
            lines.append(f"{icon} *{r['name']}* — {status_text}")
            lines.append(f"   Attempt: *#{r['attempt']}*  |  Log lines: {r['log_lines']}")
            lines.append(f"```\n{r['last_lines']}\n```\n")

    await msg.edit_text("\n".join(lines), parse_mode="Markdown")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await update.message.reply_text(
        "🤖 *Oracle ARM Retry Monitor*\n\n"
        "/status — Check retry script status on all VMs\n"
        "/id — Get your Telegram chat ID",
        parse_mode="Markdown",
    )


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /id command — return user's chat ID."""
    await update.message.reply_text(f"Your chat ID: `{update.effective_chat.id}`", parse_mode="Markdown")


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("id", cmd_id))
    logger.info("Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
