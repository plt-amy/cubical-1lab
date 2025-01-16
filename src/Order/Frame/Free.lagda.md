<!--
```agda
{-# OPTIONS --lossy-unification #-}
open import Cat.Functor.Subcategory
open import Cat.Functor.Adjoint
open import Cat.Prelude

open import Data.Bool

open import Order.Instances.Lower.Cocompletion
open import Order.Instances.Pointwise
open import Order.Semilattice.Meet
open import Order.Instances.Lower
open import Order.Diagram.Meet
open import Order.Diagram.Glb
open import Order.Diagram.Lub
open import Order.Frame
open import Order.Base

import Order.Semilattice.Meet.Reasoning as Meet-slat
import Order.Frame.Reasoning as Frm
import Order.Reasoning
```
-->

```agda
module Order.Frame.Free where
```

# Free cocompletions

A [[frame]] is, in particular, a [[meet semilattice]]. Frame
homomorphisms preserve finite meets, so they are also homomorphisms of
the underlying meet semilattices. Since equality of homomorphisms is
determined by equality of the underlying functions, these remarks
assemble into a functor $\thecat{Frames} \to \thecat{SLat}$.

<!--
```agda
open Functor
open Subcat-hom
open is-frame-hom
```
-->

```agda
Frame↪SLat : ∀ {o ℓ} → Functor (Frames o ℓ) (Meet-slats o ℓ)
Frame↪SLat .F₀ (_ , A) = Frm.meets A

Frame↪SLat .F₁ f .hom = f .hom
Frame↪SLat .F₁ f .witness = has-meet-slat-hom (f .witness)
Frame↪SLat .F-id    = trivial!
Frame↪SLat .F-∘ f g = trivial!
```

The question this module seeks to answer is: is there a way to freely
turn a semilattice $A$ into a frame $D(A)$, such that the meets in
$D(A)$ are preserved by the inclusion $A \to D(A)$? And if so, how?

Fortunately the answer is positive: otherwise this module would be much
longer (and probably rely on silly cardinality arguments). The free
frame on a semilattice is the free _cocompletion_ of that semilattice,
its [poset of lower sets][low]. The lower sets of _any_ poset are a
frame, because meets and joins are computed pointwise in the [poset of
propositions], $\Omega$.

[poset of propositions]: Order.Instances.Props.html
[low]: Order.Instances.Lower.html

```agda
Lower-sets-frame : ∀ {o ℓ} → Meet-semilattice o ℓ → Frame (o ⊔ ℓ) o
Lower-sets-frame (P , L) = Lower-sets P , L↓-frame where
  module L = Meet-slat L
  module L↓ = Order.Reasoning (Lower-sets P)

  L↓-frame : is-frame (Lower-sets P)
  L↓-frame .is-frame._∩_ a b = Lower-sets-meets P a b .Meet.glb
  L↓-frame .is-frame.∩-meets a b = Lower-sets-meets P a b .Meet.has-meet
  L↓-frame .is-frame.has-top = Lower-sets-top P
  L↓-frame .is-frame.⋃ k = Lower-sets-cocomplete P k .Lub.lub
  L↓-frame .is-frame.⋃-lubs k = Lower-sets-cocomplete P k .Lub.has-lub
  L↓-frame .is-frame.⋃-distribl x f = ext λ arg → biimp
    (rec! λ x≤a i fi≤a → inc (i , x≤a , fi≤a))
    (rec! λ i x≤a fi≤a → x≤a , inc (i , fi≤a))
```

The unit map $A \to D(A)$ sends an element of $A$ to its down-set,
$\darr A$. By a decategorification of the similar arguments about the
[Yoneda embedding], the map $A \mapsto \darr A$ preserves finite meets
(really, it preserves _all_ glbs that exist in $A$); and for a complete
lattice $B$, the [[left Kan extension]] $\Lan_{\darr}(f) : D(A) \to B$ of
a map $f : A \to B$ along $\darr$ is always cocontinuous, and left exact
whenever $f$ is. Since we're extending semilattice homomorphisms, this
means that $\Lan_{\darr}(f)$ is a frame homomorphism.

[Yoneda embedding]: Cat.Functor.Hom.html#the-yoneda-embedding

Note the similarity between the construction of free frames outlined in
the paragraph above and [Diaconescu's theorem]: “A map of frames $D(A)
\to B$ corresponds to a lex monotone map $A \to B$” is precisely the
decategorification of “A geometric morphism $B \to D(A)$ is a flat
functor $A \to B$”.

[Diaconescu's theorem]: Topoi.Classifying.Diaconescu.html

```agda
make-free-cocompletion : ∀ {ℓ} → (A : Meet-semilattice ℓ ℓ) → Free-object Frame↪SLat A
make-free-cocompletion {ℓ} A = go where
```

Anwyay, that was a _very_ dense explanation of the universal property.
Let's go through it again, this time commenting on everything as we
encounter it. We start by packaging together the extension of a
semilattice homomorphism $A \to B$ to a frame homomorphism $DA \to B$.

```agda
  module A  = Meet-slat (A .snd)

  module Mk (B : Frame ℓ ℓ)
            (f : Meet-slats.Hom A (Frm.meets (B .snd)))
    where
    module A↓ = Frm (Lower-sets-frame A .snd)
    module B  = Frm (B .snd)
    module f = is-meet-slat-hom (f .witness)
```

The easy part is an appeal to the existing machinery for free
cocompletions: Any monotone map $A \to B$ extends to a _cocontinuous_
map $DA \to B$, because $B$, being a frame, is cocomplete.

```agda
    mkhom : Frames.Hom (Lower-sets-frame A) B
    mkhom .hom = Lan↓ B.⋃-lubs (f .hom)
    mkhom .witness .⋃-≤ g = B.≤-refl' $
      Lan↓-cocontinuous B.⋃-lubs (f .hom) g
```

The harder part is showing that the cocontinuous extension of a
semilattice homomorphism is still a semilattice homomorphism. It
preserves the top element, since the cocontinuous extension takes
$\top_A$ to $\int_{B} \top_A$, which is readily calculated to
equal $\top_B$:

```agda
    mkhom .witness .top-≤ =
      B.top                   B.≤⟨ f.top-≤ ⟩
      f # A.top               B.≤⟨ B.⋃-inj (A.top , tt) ⟩
      B.⋃ (λ i → f # fst i)   B.≤∎
```

Slightly harder, but still a bit of algebra, is computing that binary
meets are preserved as well. The first step follows from the infinite
distributive law in $B$, and the second from the fact that $f$ preserves
binary meets.

```agda
    mkhom .witness .∩-≤ S T =
      B.⋃ (λ i → f # fst i) B.∩ B.⋃ (λ i → f # fst i)   B.=⟨ B.⋃-∩-product (λ i → hom f # fst i) (λ i → hom f # fst i) ⟩
      B.⋃ (λ i → f # fst (fst i) B.∩ f # fst (snd i))   B.≤⟨ B.⋃≤⋃-over meet-section (λ i → f.∩-≤ _ _) ⟩
      B.⋃ (λ i → f # fst i)                             B.≤∎
      where
        meet-section : ∫ₚ S × ∫ₚ T → ∫ₚ (λ x → x ∈ S × x ∈ T)
        meet-section ((x , p) , (y , q)) =
          x A.∩ y , S .pres-≤ A.∩≤l p , T .pres-≤ A.∩≤r q
```

It's also free from the definition of cocompletions that the extended
map $\widehat{f}$ satisfies $\widehat{f}(\darr x) = f(x)$.

```agda
    mkcomm : ∀ x → mkhom # (↓ (A .fst) x) ≡ f # x
    mkcomm x =
      (Lan↓-commutes B.⋃-lubs (f .hom) x)
```

Now we must define the unit map. We've already committed to defining
$\eta_a = \darr a$, so we have to show that $\darr$ preserves finite
meets. This is true because $\darr$ lands in _lower sets_, so it
suffices to show an equivalence between (e.g.) being _under $a$ and
under $b$_ and being _under $a \land b$_. But that's the definition of
$\land$.

```agda
  the-unit
    : (S : Meet-semilattice ℓ ℓ)
    → Meet-slats.Hom S (Frm.meets (Lower-sets-frame S .snd))
  the-unit S = go where
    module S = Meet-slat (S .snd)
    module S↓ = Frm (Lower-sets-frame S .snd)
    go : Meet-slats.Hom S S↓.meets
    go .hom = よₚ (S .fst)
    go .witness .is-meet-slat-hom.∩-≤ x y z (p , q) = do
      z≤x ← p
      z≤y ← q
      pure $ S.∩-universal z z≤x z≤y
    go .witness .is-meet-slat-hom.top-≤ x _ = pure S.!
```

We're already 80% done with the adjunction. The final thing to do is to
put it all together, bringing in the result about uniqueness of
cocontinuous extensions to tie everything up:

<!--
```agda
  open Free-object
```
-->

```agda
  go : Free-object Frame↪SLat A
  go .free = Lower-sets-frame A
  go .unit = the-unit A
  go .fold {B} f = Mk.mkhom B f
  go .commute {B} {f} = ext (Mk.mkcomm B f)
  go .unique {B} {f} g wit = ext (p #ₚ_) where
    open Mk B f

    gᵐ : Monotone (Lower-sets (A .fst)) (B .fst)
    gᵐ .hom x = g # x
    gᵐ .pres-≤ {x} {y} w = g .hom .pres-≤ w

    p = Lan↓-unique B.⋃-lubs (f .hom) gᵐ
      (is-frame-hom.pres-⋃ (g .witness))
      (wit #ₚ_)
```
