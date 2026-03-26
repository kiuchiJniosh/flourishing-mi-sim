# Maintainers

このディレクトリは monorepo から public repo へ `git subtree` で同期します。

## Source Of Truth

- package source: `src/mi_sim/`
- public subtree root: `sharing/mi_sim_public/`

## Sync From Monorepo

```bash
bash utility/sync_mi_sim_public.sh
git diff --stat -- sharing/mi_sim_public
git subtree push --prefix=sharing/mi_sim_public mi-sim-public main
```

## Initial Publish

```bash
git remote add mi-sim-public git@github.com:<org>/<repo>.git
git subtree split --prefix=sharing/mi_sim_public -b mi-sim-public-main
git push mi-sim-public mi-sim-public-main:main
```

## Pull Back Public Changes

public repo 側で直接変更した場合だけ使います。

```bash
git subtree pull --prefix=sharing/mi_sim_public mi-sim-public main --squash
```

## Safety

- `.env`, API キー, 秘密鍵, 個人情報, ログ, 生成物は含めない
- push 前に `sharing/mi_sim_public/` 配下だけを確認する
- 公開用 README は利用者向け、運用メモはこのファイルに置く
