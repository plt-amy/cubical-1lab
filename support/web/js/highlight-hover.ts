// Copyright 2002-2010, Simon Marlow.  All rights reserved.
// https://github.com/haskell/haddock/blob/ghc-8.8/LICENSE
// Slightly modified by Tesla Ice Zhang

import { Hover } from './lib/hover';

let links: Array<HTMLAnchorElement> = [];
const paths: { module: string, baseURL: string, source: string } = window as any;

const page = window.location.pathname.slice(1).replace(".html", "");

const types: Map<string, Promise<string[]>> = new Map();

/**
 * Fetch the types of identifiers used in a given module.
 *
 * @param mod The module
 * @returns
 *    A promise resolving to the types of every identifier used in that
 *    module, in some arbitrary order.
 */
async function fetchTypes(mod: string): Promise<string[]> {
  if (types.get(mod)) return types.get(mod)!;

  const prommy = fetch(`${paths.baseURL}/types/${mod}.json`, { method: 'get' }).then(async (r) => {
    if (!r.ok) throw `Failed to load type-on-hover information for module ${paths.module}`;
    return await r.json() as string[];
  });

  types.set(mod, prommy);
  return prommy;
}

/**
 * Construct a Hover appropriate for the given link element, fetching
 * the appropriate content.
 *
 * @param a The element
 * @returns The instantiated hover, or undefined if this element has no associated popup.
 */
function getHover(a: HTMLAnchorElement): Hover | undefined {
  let target;
  if (Hover.get(a)) return Hover.get(a);

  if (!Number.isNaN(target = Number.parseInt(a.getAttribute("data-identifier") ?? ""))) {
    let tgt = target;
    const mod = a.getAttribute("data-module") ?? paths.module;

    const get = fetchTypes(mod).then((tys) => {
      const element = document.createElement("div");
      element.innerHTML = tys[tgt]!;
      element.classList.add("hover-popup", "sourceCode");

      return element;
    });

    return new Hover(a, get, true);
  } else if ((target = a.getAttribute("data-target")) != null) {
    const tgt = target;
    const get = fetch(`${paths.baseURL}/fragments/${tgt}.html`, { method: 'get' }).then(async (p) => {
      if (!p.ok) throw `Failed to load fragment ${tgt}`;

      const element = document.createElement("div");
      element.innerHTML = await p.text();
      element.classList.add("hover-popup", "text-popup");

      return element;
    });

    return new Hover(a, get, false);
  }

  return;
}

/* A `highlight` event contains:
 * `link`: HTMLAnchorElement | Node
 *   either a link or a node in the dependency graph
 * `on`: boolean
 *   whether to highlight or unhighlight
 */

document.addEventListener('highlight', (({ detail: { link, on } }: CustomEvent) => {
  let match: (a : HTMLAnchorElement) => boolean;

  if (link instanceof HTMLAnchorElement) {
    match = that => that.href === link.href;

    requestAnimationFrame(async () => {
      try {
        const hover = await getHover(link);
        if (hover) {
          await hover.mouseEvent(on);
        }
      } catch (e) {
        console.log(e);
      }
    });

  } else {
    // Don't light up the entire page when hovering over the central node.
    if (link.id === page)
      return;

    match = that => that.pathname.slice(1).replace(".html", "") === link.id;
  }

  links.forEach(that => {
    if (match(that)) {
      that.classList.toggle("hover-highlight", on);
    }
  });
}) as EventListener);

export function refreshLinks(parent?: HTMLElement): HTMLAnchorElement[] {
  if (!parent) links = [];

  const here = Array.from((parent ?? document.documentElement).querySelectorAll("a[href]")) as HTMLAnchorElement[];
  links.push(...here);

  here.forEach(link => {
    link.addEventListener("mouseenter", () =>
      document.dispatchEvent(new CustomEvent('highlight', { detail: { link, on: true } })));
    link.addEventListener("mouseleave", () =>
      document.dispatchEvent(new CustomEvent('highlight', { detail: { link, on: false } })));
  });

  return here;
}

document.addEventListener("DOMContentLoaded", () => refreshLinks());

export {};
