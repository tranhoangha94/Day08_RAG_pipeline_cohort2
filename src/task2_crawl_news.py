"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    # Ví dụ:
    "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Van-ban-hop-nhat-08-VBHN-BCA-Thong-tu-lien-tich-huong-dan-ap-dung-Cac-toi-pham-ve-ma-tuy-301023.aspx",
    "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-105-2021-ND-CP-huong-dan-Luat-Phong-chong-ma-tuy-496664.aspx",
    "https://thuvienphapluat.vn/phap-luat/ho-tro-phap-luat/bo-luat-hinh-su-2015-sua-doi-2017-co-bao-nhieu-dieu-quy-dinh-ve-toi-pham-ma-tuy-do-la-nhung-dieu-na-688507-128523.html",
    "https://luatvietnam.vn/an-ninh-trat-tu/luat-phong-chong-ma-tuy-2025-so-120-2025-qh15-422329-d1.html",
    "https://thuvienphapluat.vn/phap-luat/toan-bo-khung-hinh-phat-toi-san-xuat-trai-phep-chat-ma-tuy-theo-bo-luat-hinh-su-moi-nhat-hien-nay-764729-270094.html?rel=phap_luat_chitietvb",
    # "../data/landing/legal/105_2021_ND-CP_496664.docx",
    # "../data/landing/legal/luat-phong-chong-ma-tuy-120-2025.pdf",
    # "../data/landing/legal/Toi_Pham_Ma_Tuy.docx"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    # TODO: Implement crawling logic
    # async with AsyncWebCrawler() as crawler:
    #     result = await crawler.arun(url=url)
    #     return {
    #         "url": url,
    #         "title": result.metadata.get("title", "Unknown"),
    #         "date_crawled": datetime.now().isoformat(),
    #         "content_markdown": result.markdown,
    #     }

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

    title = "Unknown"
    if result.metadata:
        title = result.metadata.get("title") or result.metadata.get("og:title") or title

    markdown = result.markdown or result.cleaned_html or ""
    if not markdown.strip():
        raise ValueError(f"Không crawl được nội dung từ: {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    saved = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ✗ Lỗi: {exc}")
            continue

        # Lưu file JSON
        saved += 1
        filename = f"article_{saved:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")

    print(f"\n✓ Tổng cộng: {saved} bài báo")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
