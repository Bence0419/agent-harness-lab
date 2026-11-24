#!/usr/bin/env bash

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is already set."
  return 0
fi

read -r -s -p "OpenAI API key: " OPENAI_API_KEY
echo

if [[ -z "$OPENAI_API_KEY" ]]; then
  echo "Error: API key cannot be empty." >&2
  unset OPENAI_API_KEY
  return 1
fi

export OPENAI_API_KEY
echo "OPENAI_API_KEY loaded into current shell session."
