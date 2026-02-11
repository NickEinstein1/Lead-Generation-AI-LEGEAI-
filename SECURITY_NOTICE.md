# 🔒 Security Notice - Credentials Removed

## ⚠️ Important Security Update

**Date**: 2026-02-10

### What Happened

The file `env` (without dot) was accidentally committed to the git repository containing sensitive credentials including:

- Database passwords (Supabase)
- JWT secret keys
- DocuSeal API keys
- Supabase anon keys
- Meta App ID and Secret

### Actions Taken

✅ **Removed from Git**: The `env` file has been removed from git tracking using `git rm --cached env`

✅ **Updated .gitignore**: Added `env` and `*.env` patterns to prevent future commits

✅ **File Still Exists Locally**: The `env` file remains on your local machine for development use

### 🚨 Required Actions

**You MUST rotate/regenerate the following credentials immediately:**

#### 1. Meta App Secret
- Go to https://developers.facebook.com/apps
- Navigate to your app → Settings → Basic
- Click "Reset App Secret"
- Update your local `env` or `.env` file with the new secret

#### 2. JWT Secret Key
- Generate a new secret:
  ```bash
  python -c "import secrets; print(secrets.token_hex(32))"
  ```
- Update `JWT_SECRET_KEY` in your `env` or `.env` file

#### 3. DocuSeal API Key
- Log in to DocuSeal
- Go to Settings → API Keys
- Revoke the old key: `mqxLjSBL9HBFGmAUXf7QwhCLQCuMHcmhhKLnfbB5PzW`
- Generate a new API key
- Update `DOCUSEAL_API_KEY` in your `env` or `.env` file

#### 4. Supabase Database Password
- Go to Supabase Dashboard
- Settings → Database → Reset Database Password
- Update `DATABASE_URL` in your `env` or `.env` file

#### 5. Supabase Anon Key (Optional but Recommended)
- The anon key is less sensitive but consider rotating it
- Supabase Dashboard → Settings → API → Generate new anon key

### 📝 Best Practices Going Forward

1. **Never commit files named**: `.env`, `env`, `.env.local`, `secrets.json`, etc.
2. **Always use**: `.env.example` for templates (without real values)
3. **Check before committing**:
   ```bash
   git status
   git diff --cached
   ```
4. **Use git hooks** to prevent accidental commits (optional)

### ✅ Current Protection

Your repository is now protected with:

```gitignore
# Environment variables
.env
env
.env.local
.env.development.local
.env.test.local
.env.production.local
*.env
```

### 🔍 Verify Protection

Check that `env` is ignored:

```bash
git status
# Should NOT show 'env' as a tracked file
```

### 📚 Additional Resources

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Git: Removing files from history](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)

---

## ⚠️ If This Repository is Public or Shared

If this repository has been pushed to GitHub, GitLab, or shared with others:

### Option 1: Remove from Git History (Recommended)

Use BFG Repo-Cleaner or git filter-branch to remove the file from all commits:

```bash
# Using BFG (easier)
brew install bfg  # or download from https://rtyley.github.io/bfg-repo-cleaner/
bfg --delete-files env
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

### Option 2: Notify Team

If you can't rewrite history (shared repository):

1. ✅ Rotate ALL credentials immediately
2. ✅ Notify all team members
3. ✅ Monitor for unauthorized access
4. ✅ Review access logs

---

## 📋 Checklist

- [ ] Removed `env` from git tracking (`git rm --cached env`)
- [ ] Updated `.gitignore` to include `env` and `*.env`
- [ ] Rotated Meta App Secret
- [ ] Generated new JWT Secret Key
- [ ] Rotated DocuSeal API Key
- [ ] Reset Supabase Database Password
- [ ] (Optional) Rotated Supabase Anon Key
- [ ] Updated local `env` or `.env` file with new credentials
- [ ] Committed changes to `.gitignore`
- [ ] (If public) Removed file from git history
- [ ] (If shared) Notified team members

---

**Status**: 🟡 In Progress - Credentials need to be rotated

**Next Step**: Rotate all credentials listed above and update your local environment file.

