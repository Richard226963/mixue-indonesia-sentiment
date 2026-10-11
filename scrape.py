import getpass
import os
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import instaloader
import pandas as pd

ACCOUNTS = ["mixueindonesia"]           # official brand accounts, the brief's "target accounts"
HASHTAGS = ["mixueindonesia", "mixue"]  # fan posts under hashtags, smaller comment sections
ACCOUNT_POSTS = 40
HASHTAG_POSTS = 50
MAX_COMMENTS_PER_POST = 100
SLEEP = 3
OUTPUT_DIR = Path("data")

PROFILE_IDS = {}  # username -> id, harvested from post owners while scraping


def login():
    user = (os.environ.get("IG_USERNAME") or input("Instagram username: ")).strip()
    if not user:
        raise SystemExit("Username Instagram wajib diisi.")

    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
        storyitem_metadata_txt_pattern="",
        max_connection_attempts=8,
        request_timeout=30,
    )

    try:
        L.load_session_from_file(user)

        if (L.test_login() or "").lower() == user.lower():
            print("using saved Instagram session")
            return L

        print("saved session expired, logging in again")

    except FileNotFoundError:
        print("no saved session, logging in")

    except instaloader.exceptions.LoginException:
        print("saved session invalid, logging in again")

    pw = os.environ.get("IG_PASSWORD") or getpass.getpass("Instagram password: ")

    try:
        try:
            L.login(user, pw)
        except instaloader.exceptions.TwoFactorAuthRequiredException:
            print(
                "Instagram meminta kode 2FA. Program ini tidak menyediakan "
                "pilihan SMS/WhatsApp atau kirim ulang kode.\n"
                "Jika kode tidak masuk, tekan Enter untuk petunjuk login browser."
            )
            code = input("2FA code (Enter jika kode tidak masuk): ").strip()
            if not code:
                raise SystemExit(
                    "Login ke instagram.com di browser komputer dan selesaikan "
                    "verifikasi SMS/WhatsApp di sana.\n"
                    "Setelah berhasil login, jalankan (contoh Chrome):\n"
                    "  uv run --link-mode=copy instaloader --load-cookies chrome\n"
                    "Lalu jalankan kembali:\n"
                    "  uv run --link-mode=copy scrapping\n"
                    "Pilihan browser dan kendala Windows dijelaskan di README.md."
                )
            try:
                L.two_factor_login(code)
            except instaloader.exceptions.BadCredentialsException:
                raise SystemExit(
                    "Kode 2FA ditolak oleh Instagram. Ini bukan pesan password salah. "
                    "Coba kode terbaru untuk login ini atau gunakan sesi browser "
                    "sesuai README.md."
                ) from None

    except instaloader.exceptions.BadCredentialsException:
        raise SystemExit("wrong username or password")

    except instaloader.exceptions.LoginException as e:
        raise SystemExit(
            f"\nInstagram rejected the login:\n{e}\n\n"
            "Complete the Instagram checkpoint in your browser, "
            "then run the script again."
        )

    L.save_session_to_file()
    print("login successful, session saved")

    return L
def post_row(post):
    return {
        "shortcode": post.shortcode,
        "username": post.owner_username,
        "posted_at_utc": post.date_utc.replace(tzinfo=timezone.utc).isoformat(),
        "caption": post.caption or "",
        "likes": post.likes,
        "comments": post.comments,
        "url": f"https://www.instagram.com/p/{post.shortcode}/",
        "is_video": post.is_video,
        "hashtags": " ".join(f"#{t}" for t in post.caption_hashtags),
    }


def save_atomic(df, path):
    # keep the previous csv intact if a later write fails
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def pull_comments(post, comment_rows, comment_path):
    # top level comments plus the replies under them; the back and forth in the
    # replies is often where the actual opinions sit
    before = len(comment_rows)
    try:
        for c in islice(post.get_comments(), MAX_COMMENTS_PER_POST):
            comment_rows.append({
                "comment_id": c.id,
                "shortcode": post.shortcode,
                "post_owner": post.owner_username,
                "username": c.owner.username if c.owner else "",
                "comment": c.text,
                "likes": c.likes_count,
                "date": c.created_at_utc.replace(tzinfo=timezone.utc).isoformat() if c.created_at_utc else "",
            })
            try:
                for a in islice(c.answers, MAX_COMMENTS_PER_POST):
                    comment_rows.append({
                        "comment_id": a.id,
                        "shortcode": post.shortcode,
                        "post_owner": post.owner_username,
                        "username": a.owner.username if a.owner else "",
                        "comment": a.text,
                        "likes": a.likes_count,
                        "date": a.created_at_utc.replace(tzinfo=timezone.utc).isoformat() if a.created_at_utc else "",
                    })
            except instaloader.exceptions.InstaloaderException:
                pass  # replies are a bonus, never let them kill the run
        if len(comment_rows) > before:
            save_atomic(pd.DataFrame(comment_rows).drop_duplicates("comment_id"), comment_path)
    except instaloader.exceptions.InstaloaderException as e:
        print(f"  komentar {post.shortcode} dilewati: {type(e).__name__}", flush=True)
    except KeyError as e:
        # instagram occasionally returns a post shape without the comment block
        print(f"  komentar {post.shortcode} dilewati: format tak terduga ({e})", flush=True)


def account_posts(loader, account):
    # instagram rate limits the profile endpoint per session sometimes, so there
    # are three ways in: the normal lookup, search, or the id we saw on a hashtag
    # post (hashtags are scraped first exactly for this)
    account = account.strip().lstrip("@").lower()
    try:
        return instaloader.Profile.from_username(loader.context, account).get_posts()
    except instaloader.exceptions.ConnectionException as e:
        print(f"  lookup langsung '{account}' dibatasi instagram, coba jalur lain "
              f"({type(e).__name__})", flush=True)
    try:
        for p in instaloader.TopSearchResults(loader.context, account).get_profiles():
            if p.username.lower() == account:
                return p.get_posts()
    except instaloader.exceptions.InstaloaderException:
        pass
    if account in PROFILE_IDS:
        node = {"id": str(PROFILE_IDS[account]), "username": account}
        return instaloader.Profile(loader.context, node).get_posts()
    raise instaloader.exceptions.ConnectionException(
        f"profil '{account}' dibatasi 429 dan tidak ada id cadangan dari hashtag; "
        "tunggu beberapa jam lalu jalankan lagi")


def hashtag_posts(loader, tag):
    tag = tag.strip().lstrip("#").lower()
    hashtag = instaloader.Hashtag.from_name(loader.context, tag)
    return hashtag.get_posts_resumable()


def scrape_source(loader, name, make_posts, max_posts):
    # one output pair (posts + comments csv) per source; each post is written
    # atomically so an interrupted run keeps whatever it already got
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    post_path = OUTPUT_DIR / f"{name}_{stamp}.csv"
    comment_path = OUTPUT_DIR / f"{name}_comments_{stamp}.csv"
    rows, comment_rows, seen = [], [], set()
    completed = False
    print(f"Mengambil maksimal {max_posts} postingan '{name}'...", flush=True)

    try:
        for post in islice(make_posts(loader), max_posts):
            if post.shortcode in seen:
                continue
            row = post_row(post)
            rows.append(row)
            seen.add(post.shortcode)
            try:
                PROFILE_IDS.setdefault(row["username"].lower(), post.owner_id)
            except instaloader.exceptions.InstaloaderException:
                pass
            save_atomic(pd.DataFrame(rows), post_path)
            pull_comments(post, comment_rows, comment_path)
            print(f"  [{len(rows)}/{max_posts}] {row['url']} ({len(comment_rows)} komentar)", flush=True)
            if len(rows) < max_posts:
                time.sleep(SLEEP)
        completed = True
    finally:
        if rows:
            status = "Selesai" if completed else "Proses terhenti; hasil sementara"
            print(f"{status}: {len(rows)} postingan, {len(comment_rows)} komentar -> {comment_path.resolve()}", flush=True)

    return len(rows), len(comment_rows)


def main() -> None:
    try:
        loader = login()
        total_posts, total_comments = 0, 0
        try:
            # hashtags first: their posts carry the owner ids the account route
            # falls back on when the profile endpoint is throttled
            jobs = [(h, lambda L, h=h: hashtag_posts(L, h), HASHTAG_POSTS) for h in HASHTAGS]
            jobs += [(a, lambda L, a=a: account_posts(L, a), ACCOUNT_POSTS) for a in ACCOUNTS]
            for name, make_posts, cap in jobs:
                try:
                    posts, comments = scrape_source(loader, name, make_posts, cap)
                    total_posts += posts
                    total_comments += comments
                except instaloader.exceptions.InstaloaderException as error:
                    print(f"sumber '{name}' dilewati: {type(error).__name__}", flush=True)
            print(f"\nTOTAL: {total_posts} postingan, {total_comments} komentar tersimpan di data/")
            if total_comments < 100:
                print("komentar masih sedikit? tambahkan akun/hashtag di daftar ACCOUNTS/HASHTAGS "
                      "atau naikkan cap postingan, lalu jalankan lagi (duplikat otomatis dibuang "
                      "saat clean.py).")
        finally:
            loader.close()
    except EOFError:
        raise SystemExit(
            "Login membutuhkan input di terminal. Jalankan uv run scrapping "
            "di PowerShell lalu masukkan username dan password Instagram di sana."
        ) from None
    except KeyboardInterrupt:
        raise SystemExit("Proses dibatalkan.") from None
    except instaloader.exceptions.InstaloaderException as error:
        raise SystemExit(
            f"Instagram tidak berhasil diakses: {error}\n"
            "Periksa koneksi dan login Instagram. Jika diminta checkpoint, "
            "selesaikan di browser. Jika akses dibatasi, tunggu sebelum mencoba lagi."
        ) from None
    except (KeyError, TypeError) as error:
        raise SystemExit(
            f"Respons Instagram tidak sesuai format yang diharapkan: {error}\n"
            "Endpoint hashtag mungkin berubah atau tidak tersedia untuk sesi ini."
        ) from None
    except OSError as error:
        raise SystemExit(f"Gagal membaca atau menyimpan file: {error}") from None


if __name__ == "__main__":
    main()