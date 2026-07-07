/**
 * <blog-carousel> — Lightweight scroll-snap carousel Web Component.
 *
 * No Shadow DOM — uses scoped `bc-` class prefixes so page CSS still
 * applies to slide content (benchmark charts, videos, images, etc.).
 *
 * The component fully owns video play/pause — do NOT put `autoplay`
 * on <video> tags.  Instead add `muted loop playsinline preload="auto"`.
 *
 * Usage:
 *   <blog-carousel loop="true">
 *     <ul>
 *       <li data-caption="Chart slide">…</li>
 *       <li data-caption="Video slide">
 *         <div class="my-wrapper">
 *           <video muted loop playsinline preload="auto">
 *             <source src="demo.mp4" type="video/mp4">
 *           </video>
 *         </div>
 *       </li>
 *     </ul>
 *   </blog-carousel>
 */
class BlogCarousel extends HTMLElement {
  static get observedAttributes() {
    return ["loop"];
  }

  constructor() {
    super();
    this._index = 0;
    this._count = 0;
    this._scrollTimer = null;
    this._isScrolling = false; // guards against scroll-handler race
  }

  /* ── Lifecycle ───────────────────────────────────────── */

  connectedCallback() {
    BlogCarousel._injectStyles();

    this._ul = this.querySelector("ul");
    this._slides = Array.from(this.querySelectorAll("ul > li"));
    this._count = this._slides.length;
    if (!this._count) return;

    /* Mark up existing DOM */
    this.classList.add("bc-root");
    this._ul.classList.add("bc-track");
    this._slides.forEach((li) => li.classList.add("bc-slide"));

    /* Strip any `autoplay` the author left on videos — we manage play/pause */
    this._slides.forEach((li) => {
      const v = li.querySelector("video");
      if (v) {
        v.removeAttribute("autoplay");
        v.pause();
      }
    });

    /* Build controls (appended after the <ul>) */
    this._buildNav();
    this._buildDots();
    this._buildCaption();
    this._buildLiveRegion();
    this._initVideoToggles();

    this._bind();
    this._syncState();
  }

  disconnectedCallback() {
    this._ul?.removeEventListener("scroll", this._onScroll);
  }

  attributeChangedCallback() {
    this._syncState();
  }

  get loop() {
    return this.getAttribute("loop") === "true";
  }

  /* ── One-time global stylesheet injection ────────────── */

  static _injected = false;

  static _injectStyles() {
    if (BlogCarousel._injected) return;
    BlogCarousel._injected = true;

    const style = document.createElement("style");
    style.id = "bc-carousel-styles";
    style.textContent = /* css */ `
      /* ── Root ──────────────────────────────────── */
      .bc-root {
        display: block;
        position: relative;
      }

      /* ── Scroll track (the <ul>) ───────────────── */
      .bc-track {
        display: flex;
        gap: 20px;
        overflow-x: auto;
        scroll-snap-type: x mandatory;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
        list-style: none;
        margin: 0;
        padding: 0;
      }
      .bc-track::-webkit-scrollbar { display: none; }

      /* ── Each slide (the <li>) ─────────────────── */
      .bc-slide {
        flex: 0 0 100%;
        scroll-snap-align: center;
        min-width: 0;
      }

      /* ── Prev / Next buttons ───────────────────── */
      .bc-btn {
        position: absolute;
        top: 0;
        z-index: 4;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        border: 2px solid #1e293b;
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        color: #1e293b;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: opacity 0.2s, background 0.2s, box-shadow 0.2s;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding: 0;
      }
      .bc-btn:hover { background: #fff; box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
      .bc-btn:active { transform: translateY(-50%) scale(0.95); }
      .bc-btn[aria-disabled="true"] {
        opacity: 0.25;
        cursor: default;
        pointer-events: none;
      }
      .bc-prev { left: 12px; }
      .bc-next { right: 12px; }

      /* ── Dots ──────────────────────────────────── */
      .bc-dots {
        display: flex;
        justify-content: center;
        gap: 8px;
        margin-top: 16px;
      }
      .bc-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        border: 2px solid #1e293b;
        background: transparent;
        cursor: pointer;
        padding: 0;
        transition: background 0.25s, transform 0.25s;
      }
      .bc-dot:hover { background: #475569; }
      .bc-dot[aria-current="true"] {
        background: #1e293b;
        transform: scale(1.2);
      }

      /* ── Caption ───────────────────────────────── */
      .bc-caption {
        text-align: center;
        font-style: italic;
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 12px;
        min-height: 1.4em;
        transition: opacity 0.3s;
      }

      /* ── Video toggle ──────────────────────────── */
      .bc-video-toggle {
        position: absolute;
        bottom: 12px;
        right: 12px;
        z-index: 3;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border: 2px solid rgba(255,255,255,0.6);
        background: rgba(0,0,0,0.45);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        color: #fff;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s;
        padding: 0;
      }
      .bc-video-toggle:hover { background: rgba(0,0,0,0.65); }

      /* ── Screen-reader live region ─────────────── */
      .bc-sr-live {
        position: absolute;
        width: 1px; height: 1px;
        overflow: hidden;
        clip: rect(0,0,0,0);
        white-space: nowrap;
      }
    `;
    document.head.appendChild(style);
  }

  /* ── Build controls ──────────────────────────────────── */

  _buildNav() {
    const makeSVG = (points) =>
      `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="${points}"/></svg>`;

    this._prevBtn = document.createElement("button");
    this._prevBtn.className = "bc-btn bc-prev";
    this._prevBtn.setAttribute("aria-label", "Previous slide");
    this._prevBtn.innerHTML = makeSVG("13 4 7 10 13 16");

    this._nextBtn = document.createElement("button");
    this._nextBtn.className = "bc-btn bc-next";
    this._nextBtn.setAttribute("aria-label", "Next slide");
    this._nextBtn.innerHTML = makeSVG("7 4 13 10 7 16");

    this.appendChild(this._prevBtn);
    this.appendChild(this._nextBtn);
  }

  _buildDots() {
    this._dotsWrap = document.createElement("div");
    this._dotsWrap.className = "bc-dots";
    this._dotsWrap.setAttribute("role", "tablist");
    this._dotsWrap.setAttribute("aria-label", "Slide navigation");

    this._dots = [];
    for (let i = 0; i < this._count; i++) {
      const dot = document.createElement("button");
      dot.className = "bc-dot";
      dot.setAttribute("role", "tab");
      dot.setAttribute("aria-label", `Go to slide ${i + 1}`);
      dot.addEventListener("click", () => this._goTo(i));
      this._dotsWrap.appendChild(dot);
      this._dots.push(dot);
    }
    this.appendChild(this._dotsWrap);
  }

  _buildCaption() {
    this._caption = document.createElement("figcaption");
    this._caption.className = "bc-caption";
    this.appendChild(this._caption);
  }

  _buildLiveRegion() {
    this._live = document.createElement("div");
    this._live.className = "bc-sr-live";
    this._live.setAttribute("aria-live", "polite");
    this._live.setAttribute("aria-atomic", "true");
    this.appendChild(this._live);
  }

  /* ── Events ──────────────────────────────────────────── */

  _bind() {
    this._prevBtn.addEventListener("click", () => this._prev());
    this._nextBtn.addEventListener("click", () => this._next());

    this._onScroll = () => {
      /* While a programmatic _goTo scroll is in flight, ignore events */
      if (this._isScrolling) return;
      clearTimeout(this._scrollTimer);
      this._scrollTimer = setTimeout(() => this._onScrollEnd(), 100);
    };
    this._ul.addEventListener("scroll", this._onScroll, { passive: true });
  }

  _onScrollEnd() {
    const scrollLeft = this._ul.scrollLeft;
    const slideW = this._slides[0].offsetWidth + 20; /* gap */
    const idx = Math.round(scrollLeft / slideW);
    const clamped = Math.max(0, Math.min(idx, this._count - 1));
    if (clamped !== this._index) {
      this._index = clamped;
      this._syncState();
    }
  }

  /* ── Navigation ──────────────────────────────────────── */

  _prev() {
    if (this._index > 0) {
      this._goTo(this._index - 1);
    } else if (this.loop) {
      this._goTo(this._count - 1);
    }
  }

  _next() {
    if (this._index < this._count - 1) {
      this._goTo(this._index + 1);
    } else if (this.loop) {
      this._goTo(0);
    }
  }

  _goTo(i) {
    this._index = i;
    const slideW = this._slides[0].offsetWidth + 20;

    /* Block the scroll handler while the programmatic scroll is animating */
    this._isScrolling = true;
    this._ul.scrollTo({ left: slideW * i, behavior: "smooth" });

    /* Release the guard after animation settles (smooth scroll ~400-500ms) */
    clearTimeout(this._scrollGuard);
    this._scrollGuard = setTimeout(() => {
      this._isScrolling = false;
    }, 600);

    this._syncState();
  }

  /* ── Sync UI state ───────────────────────────────────── */

  _syncState() {
    if (this._count === 0) return;
    const i = this._index;

    /* Prev / Next */
    if (!this.loop) {
      this._prevBtn.setAttribute("aria-disabled", i === 0 ? "true" : "false");
      this._nextBtn.setAttribute("aria-disabled", i === this._count - 1 ? "true" : "false");
    } else {
      this._prevBtn.removeAttribute("aria-disabled");
      this._nextBtn.removeAttribute("aria-disabled");
    }

    /* Position buttons vertically centered over the track */
    const trackH = this._ul.offsetHeight;
    const btnY = trackH / 2 - 22; /* half button height */
    this._prevBtn.style.top = btnY + "px";
    this._nextBtn.style.top = btnY + "px";

    /* Dots */
    this._dots?.forEach((d, j) => {
      d.setAttribute("aria-current", j === i ? "true" : "false");
    });

    /* Caption */
    const cap = this._slides[i]?.dataset.caption || "";
    this._caption.textContent = cap;

    /* Live region */
    this._live.textContent = `Slide ${i + 1} of ${this._count}${cap ? ": " + cap : ""}`;

    /* Video play/pause — active slide plays, all others pause */
    this._slides.forEach((li, j) => {
      const video = li.querySelector("video");
      if (!video) return;
      if (j === i) {
        video.currentTime = 0;
        const p = video.play();
        if (p) p.catch(() => {}); // swallow AbortError on rapid navigation
      } else {
        video.pause();
      }
    });

    /* Sync toggle button icons */
    this._slides.forEach((li, j) => {
      const video = li.querySelector("video");
      const toggle = li.querySelector(".bc-video-toggle");
      if (!video || !toggle) return;
      if (j === i) {
        toggle.innerHTML = BlogCarousel._pauseIcon();
        toggle.setAttribute("aria-label", "Pause video");
      } else {
        toggle.innerHTML = BlogCarousel._playIcon();
        toggle.setAttribute("aria-label", "Play video");
      }
    });
  }

  /* ── Video pause/play toggles ────────────────────────── */

  _initVideoToggles() {
    this._slides.forEach((li) => {
      const video = li.querySelector("video");
      if (!video) return;

      /* The slide's inner wrapper needs position:relative for the toggle */
      const wrapper = li.firstElementChild;
      if (wrapper && getComputedStyle(wrapper).position === "static") {
        wrapper.style.position = "relative";
      }

      const btn = document.createElement("button");
      btn.className = "bc-video-toggle";
      btn.setAttribute("aria-label", "Pause video");
      btn.innerHTML = BlogCarousel._pauseIcon();

      btn.addEventListener("click", () => {
        if (video.paused) {
          video.play().catch(() => {});
          btn.innerHTML = BlogCarousel._pauseIcon();
          btn.setAttribute("aria-label", "Pause video");
        } else {
          video.pause();
          btn.innerHTML = BlogCarousel._playIcon();
          btn.setAttribute("aria-label", "Play video");
        }
      });

      /* Append into the wrapper div so it overlays the video */
      (wrapper || li).appendChild(btn);
    });
  }

  static _pauseIcon() {
    return `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="4" height="12" rx="1"/><rect x="9" y="2" width="4" height="12" rx="1"/></svg>`;
  }

  static _playIcon() {
    return `<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>`;
  }
}

customElements.define("blog-carousel", BlogCarousel);
