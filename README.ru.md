# claude-max-api-proxy-rs

[English](README.md) · Русский

Локальный сервер с HTTP API OpenAI и Anthropic перед CLI `claude`. Клиент шлёт обычный запрос в `/v1/chat/completions`, прокси запускает с ним `claude --print` и возвращает ответ в том же формате.

> [!WARNING]
> Запросы оплачивает тариф Claude, под которым залогинен CLI. Anthropic [не разрешает](https://code.claude.com/docs/en/agent-sdk/overview) сторонним продуктам без её одобрения пользоваться входом через claude.ai и лимитами подписки, поэтому сторонние клиенты на тарифе Pro или Max через этот прокси ставят аккаунт под риск.

## Быстрый старт

Нужны CLI Claude Code, залогиненный в ваш аккаунт, и Rust, который ставится через [rustup.rs](https://rustup.rs).

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude auth login

git clone https://github.com/GlebSmolyakov/claude-max-api-proxy-rs.git
cd claude-max-api-proxy-rs
cargo install --path .
claude-max-api
```

Сервер слушает `127.0.0.1:8080`, только локально. Первый запрос:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Name the capital of Portugal in one word."}]}'
```

```json
{
  "id": "chatcmpl-adf63445",
  "object": "chat.completion",
  "created": 1789809523,
  "model": "claude-sonnet-5",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Lisbon" },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 477,
    "completion_tokens": 7,
    "total_tokens": 484,
    "prompt_tokens_details": { "cached_tokens": 0 }
  }
}
```

## Подключение клиента

Клиентам, совместимым с OpenAI, нужен базовый адрес `http://127.0.0.1:8080/v1`, а SDK Anthropic нужен `http://127.0.0.1:8080`. Ключи API прокси не проверяет, подойдёт любая непустая строка.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Придумай имя для функции, которая повторяет неудачные HTTP-запросы."}],
)
print(reply.choices[0].message.content)
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="local")
message = client.messages.create(
    model="sonnet",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Придумай имя для функции, которая повторяет неудачные HTTP-запросы."}],
)
print(message.content[0].text)
```

Агенты подключаются так же. [Goose](https://github.com/block/goose), например, берёт адрес из `ANTHROPIC_HOST`, а не из `ANTHROPIC_BASE_URL`:

```bash
export GOOSE_PROVIDER=anthropic GOOSE_MODEL=sonnet
export ANTHROPIC_HOST=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local
goose session
```

Как ACP-агент внутри редактора он запускается командой `goose acp` с теми же переменными.

## Эндпоинты

| Эндпоинт | Метод | Что отдаёт |
|----------|-------|------------|
| `/v1/chat/completions` | POST | OpenAI Chat Completions, целиком или стримом; `stream_options.include_usage` добавляет кусок с расходом токенов |
| `/v1/messages` | POST | Anthropic Messages, целиком или стримом |
| `/v1/models` | GET | Алиасы и все id моделей, на которых уже шли запросы, с размером контекста, когда он известен |
| `/health` | GET | Время работы, версия CLI, во что развернулся каждый алиас, ходы в ожидании результатов инструментов, расход подписки |

В ответах стоят настоящий id модели, счётчики токенов с учётом чтения и записи кэша и причина остановки; на стороне OpenAI `max_tokens` превращается в `finish_reason: "length"`, а `tool_use` в `"tool_calls"`. Ошибки приходят в формате того эндпоинта, куда шёл запрос, и с кодом, по которому видно, что случилось:

| Код | Когда |
|-----|-------|
| 400 | Запрос некорректен, имя модели неизвестно или API отклонил запрос, например не смог скачать картинку по ссылке |
| 429 | Исчерпан лимит тарифа |
| 502 | CLI упал или завершился без ответа; в конце сообщения последние строки его stderr |
| 504 | CLI 30 минут ничего не выводил |

Стриминговый ответ придерживает заголовки до первого токена, но не дольше 10 секунд. Ошибка в это окно приходит HTTP-кодом, а не событием внутри потока с кодом 200.

## Модели

| `model` в запросе | Что уходит в `claude --model` |
|-------------------|-------------------------------|
| `fable`, `opus`, `sonnet`, `haiku` | Как есть; CLI берёт самую новую модель семейства |
| Полный id: `claude-sonnet-5`, `claude-opus-5[1m]` | Как есть |
| `claude-code-cli/<имя>` | `<имя>` по тем же правилам |
| Не указана | `opus` |
| Что угодно другое, например `gpt-4o` | Ничего: прокси отвечает `400 Unknown model` |

## Что получает модель

Каждый запрос запускает CLI как голую модель: без встроенных инструментов, MCP-серверов, скиллов, файлов настроек, `CLAUDE.md` и памяти с этой машины, без фоновых вызовов при старте. Короткий запрос стоит меньше 500 входных токенов, а первый токен короткого ответа `haiku` приходит примерно через 1,5 секунды.

Системный промпт из запроса становится системным промптом CLI: это сообщения `system` и `developer` у OpenAI и поле `system` у Anthropic. Картинки доходят до модели картинками: `data:` и `http(s)`-ссылки в частях `image_url` у OpenAI, источники `base64` и `url` в блоках `image` у Anthropic. Картинки по ссылке скачивает сам API.

Полям `max_tokens`, `temperature`, `stop` и прочим параметрам генерации деваться некуда: их CLI выставляет сам.

А ещё CLI ставит короткую фиксированную преамбулу перед каждым запросом по подписке. Без собственного системного промпта модель может назвать себя агентом на Claude Agent SDK.

## Инструменты

Запрос может объявить инструменты, которые клиент выполняет сам: `tools` с записями `function` у OpenAI или `tools` с `input_schema` у Anthropic. Прокси отдаёт их CLI как MCP-сервер по адресу `/mcp/<токен>`, и модель вызывает их как свои. Когда она это делает, в ответе приходят вызовы (`tool_calls` с `finish_reason: "tool_calls"` или блоки `tool_use` со `stop_reason: "tool_use"`), а процесс CLI ждёт. Клиент выполняет инструменты и присылает результаты обычным путём, сообщениями `tool` или блоками `tool_result`. Прокси передаёт их ждущему процессу, и тот продолжает работу без перезапуска: агентная задача из четырёх шагов, например прочитать файл, поправить, показать и ответить, проходит на одном процессе CLI.

Результатов инструментов процесс ждёт до 30 минут, потом прокси его останавливает. `tool_choice: "none"` у OpenAI или `{"type": "none"}` у Anthropic прячет инструменты от модели, остальные значения `tool_choice` принимаются, но не соблюдаются. Серверные инструменты Anthropic, например веб-поиск, пропускаются.

## Диалоги

Клиенты с каждым запросом заново шлют весь разговор. Прокси хранит, какая сессия CLI содержит каждую историю, так что следующий вопрос продолжает эту сессию через `claude --resume <id> --fork-session`, и отправляется только новое сообщение. Модель видит настоящий многоходовый диалог, а API может отдавать прошлые ходы из кэша промптов.

Перегенерация одного из прошлых ответов уходит в отдельную ветку и с основной не смешивается. Если историю прокси не видел, потому что её отредактировали или она пролежала без дела больше суток, он начинает новую сессию и передаёт прошлые ходы расшифровкой, вместе с картинками. Следующий ход после этого продолжает сессию как обычно.

Карта хранится в `~/.claude-max-api/sessions.json` и переживает перезапуск. Раз в час прокси удаляет записи, которыми не пользовались сутки, вместе с их транскриптами CLI.

## Состояние и расход подписки

`/health` после нескольких запросов:

```json
{
  "status": "ok",
  "uptime": 53,
  "cli_version": "2.1.276 (Claude Code)",
  "workdir": "/Users/you/.claude-max-api/workdir",
  "saved_sessions": 9,
  "waiting_for_tools": 0,
  "models": { "haiku": "claude-haiku-4-5-20251001" },
  "rate_limits": {
    "status": "allowed",
    "windows": {
      "five_hour": { "utilization": 0.25, "resets_at": 1789824000 },
      "seven_day": { "utilization": 0.03, "resets_at": 1790334000 }
    },
    "reported_at": 1789808589
  }
}
```

`rate_limits` повторяет то, что CLI сообщил после последнего запроса. `utilization` показывает израсходованную долю каждого окна, `resets_at` время сброса в Unix-секундах. До первого запроса поле равно `null`.

## Параметры

```bash
claude-max-api [PORT] [--cwd DIR]
```

| Параметр | По умолчанию | Что задаёт |
|----------|--------------|------------|
| `PORT` | `8080` | Порт на `127.0.0.1` |
| `--cwd DIR` | `~/.claude-max-api/workdir` | Рабочую папку процессов CLI; их транскрипты CLI сохраняет в `~/.claude/projects/`, в папку с именем по этому пути |
| `RUST_LOG` | `claude_max_api=info` | Фильтр логов в синтаксисе [`EnvFilter` из tracing](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html); с `claude_max_api=debug` в лог попадает ещё и stderr CLI |

## Разработка

```bash
cargo test
cargo clippy --all-targets
```

```
src/
├── main.rs          запуск, папка состояния, остановка
├── server.rs        роутер и общее состояние
├── routes.rs        HTTP-обработчики и стриминг
├── conversation.rs  ходы и инструменты запроса, вход CLI, ключи истории
├── turn.rs          один запрос: продолжить сессию, начать заново или продолжить припаркованный ход
├── bridge.rs        MCP-сервер, через который CLI вызывает инструменты клиента
├── subprocess.rs    процесс claude и его NDJSON-вывод
├── session.rs       соответствие истории и сессии CLI
├── models.rs        допустимые имена моделей
├── status.rs        время работы, лимиты и id моделей для /health
├── error.rs         ошибки в форматах OpenAI и Anthropic
├── types/           типы сообщений OpenAI, Anthropic и CLI
└── adapter/         запросы на входе, ответы и события стрима на выходе
```

## Лицензия

MIT, текст в [LICENSE](LICENSE).
