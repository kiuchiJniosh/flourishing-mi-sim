# mi-sim

`mi_sim` は、MI ベースのカウンセラーとクライアントの self-play シミュレーションを実行するための Python package です。

## 含まれるもの

- `pyproject.toml`
- `.env.example`
- `src/mi_sim/`
- `src/mi_sim/config/`

この subtree には、公開してよい package 本体と公開向け README だけを含めます。`app/`, `utility/`, `vendor/`, `results/`, `.github/` など monorepo 固有のものは含めません。

## クイックスタート

```bash
python -m pip install -e .
cp .env.example .env  # 必要なら自分で作成
python -m mi_sim self-play --max-turns 5
```

`.env` には少なくとも `OPENAI_API_KEY` が必要です。

## maintainer メモ

monorepo 側では先に同期スクリプトを実行してから subtree を push します。

```bash
bash utility/sync_mi_sim_public.sh
git diff --stat -- sharing/mi_sim_public
git subtree push --prefix=sharing/mi_sim_public mi-sim-public main
```

初回は remote 登録後に split/push を使います。

```bash
git remote add mi-sim-public git@github.com:<org>/<repo>.git
git subtree split --prefix=sharing/mi_sim_public -b mi-sim-public-main
git push mi-sim-public mi-sim-public-main:main
```

public 側の変更を monorepo に取り込む場合だけ pull を使います。

```bash
git subtree pull --prefix=sharing/mi_sim_public mi-sim-public main --squash
```

## 注意事項

- `.env`、API キー、秘密鍵、個人情報、ログ、生成物は含めません。
- push 前に `sharing/mi_sim_public/` 配下の差分だけを確認してください。
