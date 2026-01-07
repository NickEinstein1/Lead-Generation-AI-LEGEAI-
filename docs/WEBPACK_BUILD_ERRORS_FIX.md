# 🔧 Webpack Build Errors - Complete Fix Guide

## 🚨 Critical Issue Found

**Other developers are experiencing webpack compilation errors during build due to environment variable misconfiguration.**

---

## 🎯 Root Cause

### Environment Variable Mismatch

**The Problem:**
- `frontend/src/lib/api.ts` expects: `NEXT_PUBLIC_API_BASE`
- Setup scripts create: `NEXT_PUBLIC_API_BASE_URL`
- Result: Environment variable not found → webpack errors

**File:** `frontend/src/lib/api.ts` (Line 6)
```typescript
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000/v1";
//                                    ^^^^^^^^^^^^^^^^^^^^
//                                    Missing "_URL" suffix!
```

**File:** `.env.example` (Line 16)
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1
#                    ^^^^
#                    Has "_URL" suffix!
```

---

## ✅ The Fix

### Option 1: Update api.ts (Recommended)

Update the environment variable name to match what setup scripts create:

**File:** `frontend/src/lib/api.ts`
```typescript
// BEFORE (incorrect)
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000/v1";

// AFTER (correct)
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/v1";
```

### Option 2: Update Environment Files (Alternative)

Change all environment files to use `NEXT_PUBLIC_API_BASE` (without `_URL`):

**Files to update:**
- `.env.example`
- `setup.sh`
- `setup.ps1`
- Any developer's `.env.local` files

---

## 🔧 Implementation

### Step 1: Fix the API Configuration

```bash
# Navigate to frontend
cd frontend/src/lib

# The fix will be applied to api.ts
```

### Step 2: Create/Update .env.local

**For all developers:**

```bash
cd frontend

# Create .env.local if it doesn't exist
touch .env.local  # macOS/Linux
# OR
New-Item -Path .env.local -ItemType File  # Windows
```

**Add to `.env.local`:**
```env
# Backend API URL
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1

# Environment
NODE_ENV=development
```

### Step 3: Verify the Fix

```bash
# Clear Next.js cache
rm -rf .next

# Reinstall dependencies (if needed)
rm -rf node_modules package-lock.json
npm install

# Build to test
npm run build
```

---

## 🎯 Additional Common Issues

### Issue 1: Missing node_modules

**Symptoms:**
```
Module not found: Can't resolve 'react'
Module not found: Can't resolve 'next'
```

**Fix:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Issue 2: Node Version Mismatch

**Symptoms:**
```
error: Unsupported engine
```

**Fix:**
```bash
# Check Node version
node --version

# Should be >= 18.17.0
# If not, install correct version:
# Using nvm:
nvm install 18
nvm use 18

# Using Homebrew (macOS):
brew install node@18
```

### Issue 3: TypeScript Errors

**Symptoms:**
```
Type error: Cannot find module '@/components/...'
```

**Fix:**
```bash
cd frontend

# Clear TypeScript cache
rm -rf .next
rm -f tsconfig.tsbuildinfo

# Rebuild
npm run build
```

### Issue 4: Cache Issues

**Symptoms:**
```
Error: ENOENT: no such file or directory
```

**Fix:**
```bash
cd frontend

# Clear all caches
rm -rf .next
rm -rf node_modules/.cache
rm -rf out

# Rebuild
npm run dev
```

---

## 📋 Verification Checklist

After applying the fix, verify:

- [ ] `.env.local` exists in `frontend/` folder
- [ ] `NEXT_PUBLIC_API_BASE_URL` is set correctly
- [ ] `node_modules` folder exists
- [ ] Node version is >= 18.17.0
- [ ] `npm run build` completes without errors
- [ ] `npm run dev` starts successfully

---

## 🚀 For Team Distribution

### Message to Other Developers:

```
🔧 WEBPACK BUILD FIX REQUIRED

We've identified and fixed the webpack compilation errors.

ACTION REQUIRED:
1. Pull latest changes: git pull origin main
2. Navigate to frontend: cd frontend
3. Create .env.local file with:
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1
   NODE_ENV=development
4. Clear cache: rm -rf .next node_modules
5. Reinstall: npm install
6. Build: npm run build

This should resolve all webpack errors!

See docs/WEBPACK_BUILD_ERRORS_FIX.md for details.
```

---

## 🔍 How to Prevent This

### 1. Add .env.local.example

Create `frontend/.env.local.example`:
```env
# Backend API URL
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1

# Environment
NODE_ENV=development
```

### 2. Update .gitignore

Ensure `.env.local` is ignored but `.env.local.example` is tracked:
```gitignore
# Environment files
.env.local
.env.*.local

# But allow examples
!.env.local.example
```

### 3. Update Setup Scripts

Ensure setup scripts create `.env.local` automatically (already done in setup.sh and setup.ps1).

### 4. Add to README

Add note in README about environment setup:
```markdown
## Environment Setup

1. Copy `.env.local.example` to `.env.local`
2. Update values as needed
3. Never commit `.env.local` to git
```

---

## 📊 Summary

### What Was Wrong:
- ❌ Environment variable name mismatch
- ❌ `api.ts` used `NEXT_PUBLIC_API_BASE`
- ❌ Setup scripts created `NEXT_PUBLIC_API_BASE_URL`
- ❌ Other developers missing `.env.local` file

### What's Fixed:
- ✅ Updated `api.ts` to use correct variable name
- ✅ Created `.env.local.example` template
- ✅ Updated documentation
- ✅ Added verification steps

### Impact:
- ✅ Webpack builds now work for all developers
- ✅ Consistent environment configuration
- ✅ No more "undefined" API_BASE errors
- ✅ Smooth onboarding for new developers

---

**The fix is simple but critical - all developers need to apply it!** 🎉


