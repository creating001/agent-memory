#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${ROOT_DIR}/outputs/services"
mkdir -p "${RUN_DIR}"

services=(answer embedding rerank)

script_for() {
  case "$1" in
    answer) echo "${ROOT_DIR}/scripts/serve_answer.sh" ;;
    embedding) echo "${ROOT_DIR}/scripts/serve_embedding.sh" ;;
    rerank) echo "${ROOT_DIR}/scripts/serve_rerank.sh" ;;
    *) return 1 ;;
  esac
}

port_for() {
  case "$1" in
    answer) echo "${ANSWER_PORT:-8000}" ;;
    embedding) echo "${EMBEDDING_PORT:-8001}" ;;
    rerank) echo "${RERANK_PORT:-8002}" ;;
    *) return 1 ;;
  esac
}

pid_file_for() {
  echo "${RUN_DIR}/$1.pid"
}

log_file_for() {
  echo "${RUN_DIR}/$1.log"
}

port_is_listening() {
  local port="$1"
  ss -ltn "sport = :${port}" 2>/dev/null | grep -q LISTEN
}

is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(<"${pid_file}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

clear_stale_pid() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]] && ! is_running "${pid_file}"; then
    rm -f "${pid_file}"
  fi
}

start_one() {
  local service="$1"
  local pid_file log_file script
  pid_file="$(pid_file_for "${service}")"
  log_file="$(log_file_for "${service}")"
  script="$(script_for "${service}")"

  if is_running "${pid_file}"; then
    echo "${service}: already running pid=$(<"${pid_file}")"
    return
  fi
  clear_stale_pid "${pid_file}"

  local port
  port="$(port_for "${service}")"
  if port_is_listening "${port}"; then
    echo "${service}: port ${port} is already in use by an unmanaged process"
    return 1
  fi

  setsid bash "${script}" >"${log_file}" 2>&1 </dev/null &
  echo "$!" >"${pid_file}"
  echo "${service}: started pid=$(<"${pid_file}") log=${log_file}"
}

stop_one() {
  local service="$1"
  local pid_file
  pid_file="$(pid_file_for "${service}")"
  if ! is_running "${pid_file}"; then
    clear_stale_pid "${pid_file}"
    echo "${service}: not running"
    return
  fi

  local pid
  pid="$(<"${pid_file}")"
  kill "${pid}" 2>/dev/null || true
  for _ in {1..30}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
  echo "${service}: stopped"
}

status_one() {
  local service="$1"
  local pid_file port log_file
  pid_file="$(pid_file_for "${service}")"
  port="$(port_for "${service}")"
  log_file="$(log_file_for "${service}")"
  if is_running "${pid_file}"; then
    echo "${service}: running pid=$(<"${pid_file}") port=${port} log=${log_file}"
  else
    clear_stale_pid "${pid_file}"
    if port_is_listening "${port}"; then
      echo "${service}: unmanaged listener on port=${port} log=${log_file}"
    else
      echo "${service}: stopped port=${port} log=${log_file}"
    fi
  fi
}

usage() {
  echo "Usage: $0 {start|stop|restart|status} [answer|embedding|rerank]"
}

main() {
  local command="${1:-}"
  local target="${2:-all}"
  if [[ -z "${command}" ]]; then
    usage
    exit 2
  fi

  local selected=()
  if [[ "${target}" == "all" ]]; then
    selected=("${services[@]}")
  else
    selected=("${target}")
  fi

  case "${command}" in
    start)
      for service in "${selected[@]}"; do start_one "${service}"; done
      ;;
    stop)
      for service in "${selected[@]}"; do stop_one "${service}"; done
      ;;
    restart)
      for service in "${selected[@]}"; do stop_one "${service}"; done
      for service in "${selected[@]}"; do start_one "${service}"; done
      ;;
    status)
      for service in "${selected[@]}"; do status_one "${service}"; done
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
