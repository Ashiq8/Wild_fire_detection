import re
import time
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By


# ============================================================
# SETTINGS
# ============================================================

DATASET_URL = (
    "https://ieee-dataport.org/open-access/"
    "flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs"
)

# Masks will be saved here
OUT_DIR = Path("data/masks")

# IMPORTANT:
# Masks are PNG files, e.g. image_913.png
PATTERN = re.compile(r"(image_\d+\.png)", re.I)

# First test with 10 files.
# After confirming everything works, change to:
# LIMIT = None
LIMIT = None

# Delay between downloads
DELAY = 2.0

# Number of retry attempts for a failed download
RETRIES = 3


# ============================================================
# NATURAL SORT
# ============================================================

def natural_key(name):
    """
    Sorts:
        image_1.png
        image_2.png
        image_10.png

    instead of:
        image_1.png
        image_10.png
        image_2.png
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


# ============================================================
# COLLECT MASK LINKS
# ============================================================

def collect_links(driver):
    """
    Find all PNG mask links visible on the IEEE DataPort page.
    """
    links = {}

    for a in driver.find_elements(By.TAG_NAME, "a"):
        try:
            text = a.text or ""
            href = a.get_attribute("href")

            match = PATTERN.search(text)

        except Exception:
            continue

        if match and href:
            filename = match.group(1)
            links[filename] = href

    return links


# ============================================================
# CHECK PNG FILE
# ============================================================

def is_png(path):
    """
    Check whether the downloaded file is actually a PNG.
    PNG files begin with this 8-byte signature.
    """
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


# ============================================================
# DOWNLOAD ONE MASK
# ============================================================

def download_one(session, name, url):
    """
    Download one mask file.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    destination = OUT_DIR / name

    # Masks are small (~10 KB), so DO NOT use a 100 KB threshold.
    if destination.exists() and is_png(destination):
        return "skipped (already have it)"

    temporary = destination.with_suffix(".part")

    for attempt in range(1, RETRIES + 1):

        try:
            with session.get(
                url,
                stream=True,
                timeout=60,
                allow_redirects=True
            ) as response:

                response.raise_for_status()

                with open(temporary, "wb") as f:
                    for chunk in response.iter_content(
                        chunk_size=1 << 16
                    ):
                        if chunk:
                            f.write(chunk)

            # Check that we actually received a PNG
            if not is_png(temporary):
                temporary.unlink(missing_ok=True)

                return (
                    "FAILED: downloaded file is not a PNG "
                    "(check login/session)"
                )

            # Rename .part -> .png
            temporary.replace(destination)

            size_kb = destination.stat().st_size / 1024

            return f"ok ({size_kb:.2f} KB)"

        except Exception as error:

            if attempt == RETRIES:
                temporary.unlink(missing_ok=True)

                return f"FAILED: {error}"

            print(
                f"  Attempt {attempt}/{RETRIES} failed. "
                f"Retrying..."
            )

            time.sleep(2 * attempt)


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FLAME MASK DOWNLOADER")
    print("=" * 60)

    print("\nOpening IEEE DataPort...")

    # Open Chrome
    driver = webdriver.Chrome()

    driver.get(DATASET_URL)

    print("\nIn the Chrome window:")

    print("1. Log in to IEEE DataPort.")
    print("2. Open:")
    print("      Masks_zip")
    print("          ↓")
    print("      Masks")
    print("3. Scroll through the list so the mask files are loaded.")
    print("4. Then return to this terminal.")

    input("\nPress ENTER when the mask filenames are visible... ")

    # ========================================================
    # SCAN PAGE
    # ========================================================

    while True:

        links = collect_links(driver)

        print(f"\nFound {len(links)} mask links.")

        if links:

            print("\nFirst few files found:")

            for name in sorted(
                links.keys(),
                key=natural_key
            )[:10]:

                print("  ", name)

            answer = input(
                "\nPress ENTER to continue downloading, "
                "or type 'r' to scan again: "
            )

            if answer.strip().lower() != "r":
                break

        else:

            answer = input(
                "\nNo PNG mask links found.\n"
                "Make sure you opened Masks_zip > Masks "
                "and the filenames are visible.\n\n"
                "Type 'r' to scan again or 'q' to quit: "
            )

            if answer.strip().lower() == "q":
                driver.quit()
                return


    # ========================================================
    # COPY LOGIN COOKIES TO REQUESTS SESSION
    # ========================================================

    print("\nCopying browser session...")

    session = requests.Session()

    for cookie in driver.get_cookies():

        session.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=cookie.get("domain")
        )

    # Use the same User-Agent as Chrome
    try:
        session.headers["User-Agent"] = driver.execute_script(
            "return navigator.userAgent"
        )
    except Exception:
        pass


    # ========================================================
    # SORT FILES
    # ========================================================

    names = sorted(
        links.keys(),
        key=natural_key
    )

    # LIMIT = 10 for initial test
    if LIMIT is not None:
        names = names[:LIMIT]


    print("\n" + "=" * 60)
    print(f"Files selected for download: {len(names)}")
    print("=" * 60)


    # ========================================================
    # DOWNLOAD
    # ========================================================

    failed = []

    for index, name in enumerate(names, start=1):

        url = links[name]

        result = download_one(
            session,
            name,
            url
        )

        print(
            f"[{index}/{len(names)}] "
            f"{name}: {result}"
        )

        if result.startswith("FAILED"):
            failed.append(name)

        time.sleep(DELAY)


    # ========================================================
    # SUMMARY
    # ========================================================

    successful = len(names) - len(failed)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Total selected : {len(names)}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {len(failed)}")
    print(f"Output folder  : {OUT_DIR.resolve()}")

    if failed:

        print("\nFailed files:")

        for name in failed[:20]:
            print("  ", name)

        if len(failed) > 20:
            print(
                f"  ... and {len(failed) - 20} more"
            )

    print("\nClosing Chrome...")

    driver.quit()

    print("Done.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()