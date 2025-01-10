import { refreshLinks } from "../highlight-hover";
import { Timeout } from "./timeout";

const showTimeout: number = 250;
const hideTimeout: number = 500;
const
  currentHovers: Map<HTMLElement, Hover> = new Map(),
  hovers: Map<string, Hover> = new Map();

export class Hover {
  /* To put ourselves in the `hovers` map, we need to be able to refer to
   * hovers by something that's not the element, since, in the
   * constructor, the element may not exist yet. */
  private static nextHover = 0;

  /** How many cursors are currently over this hover? */
  private cursors: number = 0;

  /** Function to be invoked when this popup disappears. Also serves as a
   * proxy for whether or not we've drawn it.
   * */
  private shown?: () => void;

  /** If this popup is associated to an element in another popup, this
   * stores that other popup. */
  private parent?: Hover;

  /** Stores any of the children of this popup that have been rendered.
   * The promise resolves when that child closes. */
  private children: Map<Hover, Promise<void>> = new Map();

  /**
   * Constructs the hover, and sets up the popup element.
   *
   * @param anchor    The link to which we belong.
   * @param element   A promise that will resolve with the popup container.
   * @param ephemeral
   *    Whether this popup should fade in/out, and also whether it
   *    should be shared.
   */
  constructor(private anchor: HTMLElement, private element: Promise<HTMLDivElement>, private ephemeral: boolean) {
    currentHovers.set(anchor, this);

    // We figure out parents by id because we don't yet have the element
    // here.
    const
      myself = `hover-popup-${++Hover.nextHover}`,
      parent = anchor.closest("div.hover-popup");

    hovers.set(myself, this);
    if (parent && hovers.get(parent.id)) this.parent = hovers.get(parent.id);

    element.then((e) => {
      e.classList.add('popup-hidden');
      e.id = myself;

      e.addEventListener("mouseenter", async () => this.fading || await this.mouseEvent(true));
      e.addEventListener("mouseleave", async () => this.fading || await this.mouseEvent(false));

      refreshLinks(e);

      document.body.appendChild(e);
    });
  }

  private fadeDirection?: 'up' | 'down';

  /**
   * Show the popup and decide on the positioning, hence in which
   * direction the popup will fade.
   */
  private async place() {
    const el = await this.element;
    if (this.cursors <= 0 || this.shown) return;

    el.classList.remove('popup-hidden');

    const selfRect = this.anchor.getBoundingClientRect();
    const hoverRect = el.getBoundingClientRect();

    if (selfRect.bottom + hoverRect.height + 48 > window.innerHeight) {
      // Tooltip placed above anchor
      el.style.top = `calc(${window.scrollY + selfRect.top - hoverRect.height}px - 1.3rem)`;
      this.fadeDirection = 'down';
    } else {
      // Tooltip placed below anchor
      el.style.top = `${window.scrollY + selfRect.bottom}px`;
      this.fadeDirection = 'up';
    }

    if (selfRect.left + hoverRect.width > window.innerWidth) {
      el.style.left = `calc(${selfRect.right - hoverRect.width}px)`;
    } else {
      el.style.left = `${selfRect.left}px`;
    }

    if (!this.ephemeral) {
      el.classList.remove(`popup-fade-out-${this.fadeDirection}`);
      el.classList.add(`popup-fade-in-${this.fadeDirection}`);
    }

    // Instead of having a boolean to indicate what's been shown, we use
    // the closing signal as a sentinel instead.
    this.shown = this.blockParent();
  }

  /**
   * Add this element to the map of its parents' children, and update
   * the closing promise.
   *
   * @returns
   *  A function that, when invoked, signals to the parent that
   *  this child has closed.
   */
  private blockParent(): () => void {
    if (!this.parent) return () => { };

    let resolve: () => void;
    this.parent.children.set(this, new Promise((res) => {
      resolve = res
    }));

    return () => {
      this.parent?.children.delete(this);
      requestAnimationFrame(resolve!);
    }
  }

  /** Are we playing the closing animation? */
  private closing?: Timeout;

  /**
   * Hide the popup. Does not remove the element from the DOM!
   */
  private async displace() {
    const el = await this.element;

    // Set ourselves to a hiding state immediately, so that if the user
    // goes back onto the link while we're closing we can just
    // immediately show again.
    if (this.shown) this.shown();
    delete this.shown;

    if (!this.ephemeral) {
      el.classList.remove(`popup-fade-in-${this.fadeDirection}`);
      el.classList.add(`popup-fade-out-${this.fadeDirection}`);
    }

    this.closing = new Timeout(this.ephemeral ? 0 : 250, `displace ${this.anchor}`);

    await this.closing.start();
    el.classList.add('popup-hidden')

    delete this.closing;

    if (this.ephemeral) return this.destroy();
  }

  private async destroy() {
    const el = await this.element;
    el.remove();

    currentHovers.delete(this.anchor);
    hovers.delete(el.id);
  }

  /** Are we closing this popup, or does it associated with an anchor
   * that belongs to a parent which is fading? */
  private get fading(): boolean {
    return !!this.closing || (!!this.parent && this.parent.fading)
  }

  /** Timeout before the popup appears. */
  private showTimer?: Timeout;

  /**
   * Handle an update when the popup is not shown yet. Depending on
   * whether or not the cursor is over this popup (or its anchor),
   * either show it, or cancel a previous show timer if one exists..
   */
  private async unhide() {
    if (this.cursors >= 1) {
      this.showTimer = new Timeout(this.ephemeral ? 0 : showTimeout, `unhide ${this.anchor}`);

      // If we're currently playing the closing animation, but there's a
      // positive number of cursors (necessarily on the anchor), then we
      // should wait and show the popup again.
      try {
        if (this.closing) await this.closing.start();

        await Promise.all([this.showTimer?.start(), this.element]);
        await this.place();

        delete this.showTimer;
      } catch (e) {
        if (e === this.showTimer?.cancelled) return;
        throw e;
      }
    } else if (this.cursors <= 0 && this.showTimer && !this.showTimer.done) {
      this.showTimer?.cancel();
      if (this.ephemeral) this.destroy();
    }
  }


  private hideTimer?: Timeout;

  /**
   * Handle an update when the popup has been shown. Depending on
   * whether the cursor is over this popup (or its anchor), either
   * hide/destroy it, or cancel a previous unshowing.
   */
  private async unshow() {
    if (this.cursors <= 0) {
      this.hideTimer = new Timeout(this.ephemeral ? 0 : hideTimeout, `unshow ${this.anchor}`);

      try {
        await Promise.all(Array.from(this.children.values()).concat(this.hideTimer.start()));

        if (this.cursors > 0) return;
        await this.displace();

      } catch (e) {
        if (e == this.hideTimer.cancelled) return;
        throw e;
      }
    } else if (this.cursors >= 1 && this.hideTimer && !this.hideTimer.done) {
      this.hideTimer?.cancel();
    }
  }

  /**
   * Handle the mouse entering/leaving either the popup or its anchor
   * element.
   *
   * @param on Did the mouse move onto the popup, or off it?
   */
  public async mouseEvent(on: boolean) {
    this.cursors = Math.max(this.cursors + (on ? 1 : -1), 0);

    if (!this.shown) {
      await this.unhide();
    } else {
      await this.unshow();
    }
  }

  public static get(a: HTMLElement): Hover | undefined {
    return currentHovers.get(a)
  }
};
