# Maintainers

This directory is synchronized from the monorepo to the public repository using `git subtree`.

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

Use this only if changes were made directly in the public repository.

```bash
git subtree pull --prefix=sharing/mi_sim_public mi-sim-public main --squash
```

## Safety

- Do not include `.env`, API keys, private keys, personal data, logs, or generated artifacts
- Before pushing, review only the contents under `sharing/mi_sim_public/`
- Keep user-facing documentation in the public `README`; keep operational notes in this file
