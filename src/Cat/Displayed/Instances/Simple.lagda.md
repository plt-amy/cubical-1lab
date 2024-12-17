<!--
```agda
open import Cat.Displayed.Instances.Slice
open import Cat.Displayed.Comprehension
open import Cat.Diagram.Product.Solver
open import Cat.Displayed.Cartesian
open import Cat.Displayed.Functor
open import Cat.Diagram.Pullback
open import Cat.Diagram.Product
open import Cat.Instances.Slice
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Reasoning
```
-->

```agda
module Cat.Displayed.Instances.Simple
  {o ℓ} (B : Precategory o ℓ)
  (has-prods : ∀ X Y → Product B X Y)
  where

open Cat.Reasoning B
open Binary-products B has-prods
```

# Simple fibrations

One reason to be interested in fibrations is that they provide an
excellent setting to study logical and type-theoretical phenomena.
When constructing models of type theories, the general pattern
is to construct a category of contexts and substitutions, and then
to study types and terms as structures over this category.
The language of [[displayed categories]] allows us to capture this situation
quite succinctly by considering these extra pieces of equipment as
being fibred over our category of contexts.

Focusing in, the language of *simple fibrations* provides us with enough
structure to study simply-typed languages that have enough structure
to represent contexts internally (i.e.: product types).

To start, we fix some base category $\cB$ with binary products.
Intuitvely, this will be some sort of category of contexts, and
context extension endows this category with products. We interpret a
type in a context to be an object $\Gamma \times X : \cB$.

Maps between types in contexts $(\Gamma \times X) \to (\Delta \times Y)$
are then given by a map $\Gamma \to \Delta$ between contexts, and a
map $\Gamma \times X \to Y$, which is meant to denote a derivation
of $Y$ from $\Gamma \times X$.

To encode this as a displayed category, we define the space of
objects over some $\Gamma$ to be simply an object of $B$.
This may seem odd, but recall that we are modeling a type theory with
enough structure to consider contexts as types: if this is not the
situation (IE: STLC without products), then we need to consider a more
[refined notion].

[refined notion]: Cat.Displayed.Instances.CT-Structure.html

For the maps, we already have the map $\Gamma \to \Delta$ as the
base morphism, so the displayed portion of the map will be the
map $\Gamma \times X \to Y$ between derivations. The identity
morphism $id : \Gamma \times Y \to Y$ ignores the context, and
derives $Y$ by using the $Y$ we already had, and is thus represented
by the second projection $\pi_2$.

Composition of morphisms is somewhat more involved, but can be derived
by playing type-tetris, as it's all a matter of golfing the types
and contexts into the correct place. The category laws are then a matter
of bashing through a bunch of nested pairings and projections, and
can be entirely automated.

```agda
Simple : Displayed B o ℓ
Simple .Displayed.Ob[_] Γ = Ob
Simple .Displayed.Hom[_] {Γ} {Δ} u X Y = Hom (Γ ⊗₀ X) Y
Simple .Displayed.Hom[_]-set _ _ _ = Hom-set (_ ⊗₀ _) _
Simple .Displayed.id' = π₂
Simple .Displayed._∘'_ {f = u} {g = v} f g = f ∘ ⟨ v ∘ π₁ , g ⟩
Simple .Displayed.idr' f =
  f ∘ ⟨ (id ∘ π₁) , π₂ ⟩ ≡⟨ products! has-prods ⟩
  f                      ∎
Simple .Displayed.idl' {f = u} f =
  π₂ ∘ ⟨ u ∘ π₁ , f ⟩ ≡⟨ products! has-prods ⟩
  f                   ∎
Simple .Displayed.assoc' {f = u} {g = v} {h = w} f g h =
  f ∘ ⟨ (v ∘ w) ∘ π₁ , g ∘ ⟨ w ∘ π₁ , h ⟩ ⟩ ≡⟨ products! has-prods ⟩
  (f ∘ ⟨ v ∘ π₁ , g ⟩) ∘ ⟨ w ∘ π₁ , h ⟩     ∎
```

# Cartesian morphisms

A morphism $f' : \Gamma \times X \to Y$ in the simple fibration is
cartesian if and only if the morphism $\langle \pi_1 , f' \rangle :
\Gamma \times X \to \Gamma \times Y$ is invertible. This means that the
[[cartesian morphisms]] are the isomorphisms of types, as we are
interpreting morphisms in the simple fibration as derivations.

We begin with the reverse direction, as it is slightly simpler to show.

```agda
⟨⟩-invertible→cartesian
  : ∀ {Γ Δ x y} {f : Hom Γ Δ} {f' : Hom (Γ ⊗₀ x) y}
  → is-invertible ⟨ π₁ , f' ⟩
  → is-cartesian Simple f f'
⟨⟩-invertible→cartesian {Γ} {Δ} {x} {y} {f} {f'} ⟨⟩-inv = cart where
  module ⟨⟩-inv = is-invertible ⟨⟩-inv
  open is-cartesian
```

Let $m : \Psi \to \Gamma$, and
$h' : \Psi \times Z \to Y$ be a pair of morphisms; we need to construct
some $m' : \Psi \times Z \to X$ that factorizes $h'$ through $f'$.

We begin by constructing the map
$\langle m \circ \pi_1 , h' \rangle : \Psi \times Z \to \Gamma \times Y$,
which we can then pre-compose with the inverse to $\langle \pi_1 , f' \rangle$
to obtain a map $\Psi \times Z \to \Gamma \times X$. Finally, we can
compose these two maps with the second projection, yielding the required
map.

```agda
  cart : is-cartesian Simple f f'
  cart .universal m h' = π₂ ∘ ⟨⟩-inv.inv ∘ ⟨ m ∘ π₁ , h' ⟩
```

Showing that this map uniquely factorises $f'$ boils down to pushing
around products and using that fact that the inverse to
$\langle \pi_1 , f' \rangle$ is, in fact, an inverse.

```agda
  cart .commutes m h' =
    f' ∘ ⟨ m ∘ π₁ , π₂ ∘ ⟨⟩-inv.inv ∘ ⟨ m ∘ π₁ , h' ⟩ ⟩ ≡˘⟨ ap₂ _∘_ refl (⟨⟩-unique (pulll (π₁-inv ⟨⟩-inv) ∙ π₁∘⟨⟩) refl) ⟩
    f' ∘ ⟨⟩-inv.inv ∘ ⟨ m ∘ π₁ , h' ⟩                   ≡⟨ pulll (π₂-inv ⟨⟩-inv) ⟩
    π₂ ∘ ⟨ m ∘ π₁ , h' ⟩                                ≡⟨ π₂∘⟨⟩ ⟩
    h'                                                  ∎
  cart .unique {m = m} {h' = h'} m' p =
    m'                                                      ≡˘⟨ π₂∘⟨⟩ ⟩
    π₂ ∘ ⟨ m ∘ π₁ , m' ⟩                                    ≡⟨ ap₂ _∘_ refl (introl ⟨⟩-inv.invr) ⟩
    π₂ ∘ (⟨⟩-inv.inv ∘ ⟨ π₁ , f' ⟩) ∘ ⟨ m ∘ π₁ , m' ⟩       ≡⟨ products! has-prods ⟩
    π₂ ∘ ⟨⟩-inv.inv ∘ ⟨ m ∘ π₁ , ⌜ f' ∘ ⟨ m ∘ π₁ , m' ⟩ ⌝ ⟩ ≡⟨ ap! p ⟩
    π₂ ∘ ⟨⟩-inv.inv ∘ ⟨ m ∘ π₁ , h' ⟩ ∎
```

On to the forward direction! Let $f : \Gamma \to \Delta$ and
$f' : \Gamma \times X \to Y$ form a cartesian map in the simple fibration.
We can construct an inverse to $\langle \pi_1 , f' \rangle$ by factorizing
the map $\pi_2 : \Gamma \times Y \to Y$, as in the following diagram:

~~~{.quiver}
\begin{tikzcd}
  Y \\
  & X && Y \\
  \Gamma \\
  & \Gamma && \Delta \\
  \arrow["{\exists ! i}", dashed, from=1-1, to=2-2]
  \arrow["{f'}", from=2-2, to=2-4]
  \arrow[lies over, from=2-2, to=4-2]
  \arrow[lies over, from=2-4, to=4-4]
  \arrow[lies over, from=1-1, to=3-1]
  \arrow["f"', curve={height=-12pt}, from=3-1, to=4-4]
  \arrow["id"', from=3-1, to=4-2]
  \arrow["f"', from=4-2, to=4-4]
  \arrow["{\pi_2}"{description}, curve={height=-12pt}, from=1-1, to=2-4]
\end{tikzcd}
~~~

```agda
cartesian→⟨⟩-invertible
  : ∀ {Γ Δ x y} {f : Hom Γ Δ} {f' : Hom (Γ ⊗₀ x) y}
  → is-cartesian Simple f f'
  → is-invertible ⟨ π₁ , f' ⟩
cartesian→⟨⟩-invertible {Γ} {Δ} {x} {y} {f} {f'} cart =
  make-invertible ⟨ π₁ , universal id π₂ ⟩
    left-inv
    right-inv
    where
      open is-cartesian cart
```

Showing that this map is a left inverse can be seen by a short computation.

```agda
      left-inv : ⟨ π₁ , f' ⟩ ∘ ⟨ π₁ , universal id π₂ ⟩ ≡ id
      left-inv =
        ⟨ π₁ , f' ⟩ ∘ ⟨ π₁ , universal id π₂ ⟩           ≡⟨ products! has-prods ⟩
        ⟨ π₁ , ⌜ f' ∘ ⟨ id ∘ π₁ ,  universal id π₂ ⟩ ⌝ ⟩ ≡⟨ ap! (commutes id π₂) ⟩
        ⟨ π₁ , π₂ ⟩                                      ≡⟨ ⟨⟩-η ⟩
        id                                               ∎
```

Showing that the constructed map is a right inverse is somewhat more
involved. The key lemma is that
$i \circ \langle \pi_1 , f' \rangle : \Gamma \times X \to X$ is equal
to $\pi_2$. To see this, consider the following diagram

~~~{.quiver}
\begin{tikzcd}
  X \\
  & Y \\
  \Gamma && X && Y \\
  & \Gamma \\
  && \Gamma && \Delta \\
  \arrow["{\exists ! i}", dashed, from=2-2, to=3-3]
  \arrow["{f'}", from=3-3, to=3-5]
  \arrow[lies over, from=3-3, to=5-3]
  \arrow[lies over, from=3-5, to=5-5]
  \arrow[lies over, from=2-2, to=4-2]
  \arrow["f"', curve={height=-12pt}, from=4-2, to=5-5]
  \arrow["id"', from=4-2, to=5-3]
  \arrow["f"', from=5-3, to=5-5]
  \arrow["{\pi_2}"{description}, curve={height=-12pt}, from=2-2, to=3-5]
  \arrow["{f'}", curve={height=-18pt}, from=1-1, to=3-5]
  \arrow["id"', from=3-1, to=4-2]
  \arrow[lies over, from=1-1, to=3-1]
  \arrow["{\langle \pi_1, f' \rangle}"{description}, from=1-1, to=2-2]
\end{tikzcd}
~~~

Note that $\pi_2$ factorizes $f'$, so it must be equal to the universal
factorisation of $f'$, as $f'$ is cartesian. Furthermore,
$i \circ \langle \pi_1 , f' \rangle$ also factorizes $f'$, which lets us
see that $i \circ \langle \pi_1 , f' \rangle = \pi_2$.

```agda
      universal-π₂-unique : f' ∘ ⟨ id ∘ π₁ , universal id π₂ ∘ ⟨ π₁ , f' ⟩ ⟩ ≡ f'
      universal-π₂-unique =
        f' ∘ ⟨ id ∘ π₁ , universal id π₂ ∘ ⟨ π₁ , f' ⟩ ⟩ ≡⟨ products! has-prods ⟩
        f' ∘ ⟨ id ∘ π₁ , universal id π₂ ⟩ ∘ ⟨ π₁ , f' ⟩ ≡⟨ pulll (commutes id π₂) ⟩
        π₂ ∘ ⟨ π₁ , f' ⟩                                 ≡⟨ π₂∘⟨⟩ ⟩
        f'                                               ∎

      universal-π₂∘f' : universal id π₂ ∘ ⟨ π₁ , f' ⟩ ≡ π₂
      universal-π₂∘f' =
        universal id π₂ ∘ ⟨ π₁ , f' ⟩ ≡⟨ unique _ universal-π₂-unique ⟩
        universal id f'               ≡˘⟨ unique _ (elimr (ap₂ ⟨_,_⟩ (idl _) refl ∙ ⟨⟩-η)) ⟩
        π₂                            ∎
```

We can then apply this lemma to see that $\langle \pi_1, i \rangle$ forms
a right inverse.

```agda
      right-inv : ⟨ π₁ ,  universal id π₂ ⟩ ∘ ⟨ π₁ , f' ⟩ ≡ id
      right-inv =
        ⟨ π₁ , universal id π₂ ⟩ ∘ ⟨ π₁ , f' ⟩ ≡⟨ products! has-prods ⟩
        ⟨ π₁ , universal id π₂ ∘ ⟨ π₁ , f' ⟩ ⟩ ≡⟨ ap₂ ⟨_,_⟩ refl universal-π₂∘f' ⟩
        ⟨ π₁ , π₂ ⟩                            ≡⟨ ⟨⟩-η ⟩
        id                                     ∎
```

# Fibration structure

As suggested by it's name, the simple fibration is a fibration.
given a map $\Gamma \to Delta$ in the base, and a $(\Delta , Y)$
upstairs, we can construct a lift by selecting $(\Gamma, Y)$ as the
corner of the lift, and then using the second projection as the lift
itself. Intuitively, this encodes a substitution of contexts: because
we are working with a simple type theory, the substitutions don't need
to touch the types, as there are no possible dependencies!

<!--
```agda
open Cartesian-fibration
open Cartesian-lift
open is-cartesian
```
-->

```agda
Simple-fibration : Cartesian-fibration Simple
Simple-fibration .has-lift f Y .x' = Y
Simple-fibration .has-lift f Y .lifting = π₂
Simple-fibration .has-lift f Y .cartesian .universal _ h = h
Simple-fibration .has-lift f Y .cartesian .commutes g h = π₂∘⟨⟩
Simple-fibration .has-lift f Y .cartesian .unique {m = g} {h' = h} h' p =
  h'                   ≡˘⟨ π₂∘⟨⟩ ⟩
  π₂ ∘ ⟨ g ∘ π₁ , h' ⟩ ≡⟨ p ⟩
  h ∎
```

# Comprehension structure

The simple fibration admits a [[fibred functor]] into the [[codomain
fibration]] that maps an object $X$ over $\Gamma$ to the projection
$\pi_1 : \Gamma \times X \to \Gamma$.

```agda
Simple→Slices
  : Vertical-functor Simple (Slices B)
Simple→Slices = func where
  open Vertical-functor
  open /-Obj
  open Slice-hom

  func : Vertical-functor _ _
  func .F₀' {x} x' = cut {domain = x ⊗₀ x'} π₁
  func .F₁' {f = f} f' = slice-hom ⟨ f ∘ π₁ , f' ⟩ (sym π₁∘⟨⟩)
  func .F-id' =
    Slice-path B $
    ⟨ id ∘ π₁ , π₂ ⟩ ≡⟨ ap₂ ⟨_,_⟩ (idl _) refl ∙ ⟨⟩-η ⟩
    id               ∎
  func .F-∘' {f = f} {g = g} {f' = f'} {g' = g'} =
    Slice-path B $
    ⟨ (f ∘ g) ∘ π₁ , f' ∘ ⟨ g ∘ π₁ , g' ⟩ ⟩ ≡⟨ products! has-prods ⟩
    ⟨ f ∘ π₁ , f' ⟩ ∘ ⟨ g ∘ π₁ , g' ⟩       ∎
```

Furthermore, this functor is fibred. The general sketch is that
[cartesian morphisms in the codomain fibration are given by pullbacks][cart-pb],
and cartesian maps in the simple fibration are given by inverses to
$\langle \pi_1 , f' \rangle$, and we can use this inverse to construct the
universal map for the pullback.

[cart-pb]: Cat.Displayed.Instances.Slice.html#cartesian-maps

```agda
Simple→Slices-fibred
  : is-vertical-fibred Simple→Slices
Simple→Slices-fibred {f = f} f' cart =
  pullback→cartesian B pb
  where
    open is-pullback

    ⟨⟩-inv : is-invertible ⟨ π₁ , f' ⟩
    ⟨⟩-inv = cartesian→⟨⟩-invertible cart

    module ⟨⟩-inv = is-invertible ⟨⟩-inv

    pb : is-pullback B π₁ f ⟨ f ∘ π₁ , f' ⟩ π₁
    pb .square = sym π₁∘⟨⟩
    pb .universal {P'} {p₁'} {p₂'} p =
      ⟨⟩-inv.inv ∘ ⟨ p₁' , π₂ ∘ p₂' ⟩
```

<details>
<summary>Showing that this map is universal involves a series of somewhat
tedious calculations, so we omit them.
</summary>

```agda
    pb .p₁∘universal {P} {p₁'} {p₂'} {p} =
      π₁ ∘ ⟨⟩-inv.inv ∘ ⟨ p₁' , π₂ ∘ p₂' ⟩ ≡⟨ pulll (π₁-inv ⟨⟩-inv) ⟩
      π₁ ∘ ⟨ p₁' , π₂ ∘ p₂' ⟩              ≡⟨ π₁∘⟨⟩ ⟩
      p₁' ∎
    pb .p₂∘universal {P} {p₁'} {p₂'} {p} =
      ⟨ f ∘ π₁ , f' ⟩ ∘ ⟨⟩-inv.inv ∘ ⟨ p₁' , π₂ ∘ p₂' ⟩                ≡⟨ pulll (⟨⟩∘ _) ⟩
      ⟨ (f ∘ π₁) ∘ ⟨⟩-inv.inv , f' ∘ ⟨⟩-inv.inv ⟩ ∘ ⟨ p₁' , π₂ ∘ p₂' ⟩ ≡⟨ ap₂ _∘_ (ap₂ ⟨_,_⟩ (pullr (π₁-inv ⟨⟩-inv)) (π₂-inv ⟨⟩-inv)) refl ⟩
      ⟨ f ∘ π₁ , π₂ ⟩ ∘ ⟨ p₁' , π₂ ∘ p₂' ⟩                             ≡⟨ products! has-prods ⟩
      ⟨ f ∘ p₁' , π₂ ∘ p₂' ⟩                                           ≡⟨ ap₂ ⟨_,_⟩ p refl ⟩
      ⟨ π₁ ∘ p₂' , π₂ ∘ p₂' ⟩                                          ≡⟨ products! has-prods ⟩
      p₂' ∎
    pb .unique {P} {p₁'} {p₂'} {p} {h'} q r =
      h'                                                   ≡⟨ insertl ⟨⟩-inv.invr ⟩
      ⟨⟩-inv.inv ∘ ⟨ π₁ , f' ⟩ ∘ h'                        ≡⟨ ap₂ _∘_ refl (⟨⟩∘ h') ⟩
      ⟨⟩-inv.inv ∘ ⟨ ⌜ π₁ ∘ h' ⌝ , f' ∘ h' ⟩               ≡⟨ ap! q ⟩
      ⟨⟩-inv.inv ∘ ⟨ p₁' , ⌜ f' ∘ h' ⌝ ⟩                   ≡⟨ ap! (pushl (sym π₂∘⟨⟩)) ⟩
      ⟨⟩-inv.inv ∘ ⟨ p₁' , π₂ ∘ ⌜ ⟨ f ∘ π₁ , f' ⟩ ∘ h' ⌝ ⟩ ≡⟨ ap! r ⟩
      ⟨⟩-inv.inv ∘ ⟨ p₁' , π₂ ∘ p₂' ⟩                      ∎
```
</details>

This yields a [comprehension structure] on the simple fibration, which
encodes the structure of a non-dependent type theory.

[comprehension structure]: Cat.Displayed.Comprehension.html

```agda
Simple-comprehension : Comprehension Simple
Simple-comprehension .Vertical-fibred-functor.vert = Simple→Slices
Simple-comprehension .Vertical-fibred-functor.F-cartesian = Simple→Slices-fibred
```
