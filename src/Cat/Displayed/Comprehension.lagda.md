<!--
```agda
open import Cat.Displayed.Cartesian.Indexing
open import Cat.Displayed.Instances.Slice
open import Cat.Displayed.Cartesian
open import Cat.Displayed.Functor
open import Cat.Diagram.Pullback
open import Cat.Diagram.Comonad
open import Cat.Displayed.Total
open import Cat.Instances.Slice
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Displayed.Reasoning
import Cat.Reasoning
```
-->

```agda
module Cat.Displayed.Comprehension
  {o ℓ o' ℓ'} {B : Precategory o ℓ}
  (E : Displayed B o' ℓ')
  where
```

<!--
```agda
open Cat.Reasoning B
open Cat.Displayed.Reasoning E
open Displayed E
open Functor
open _=>_
open Total-hom
open /-Obj
open Slice-hom
```
-->

# Comprehension categories {defines="comprehension-category"}

Fibrations provide an excellent categorical framework for studying logic
and type theory, as they give us the tools required to describe objects
in a context, and substitutions between them. However, this framework
is missing a key ingredient: we have no way to describe context extension!

Before giving a definition, it is worth pondering what context extension
*does*. Consider some context $\Gamma$, and type
$\Gamma \vdash A\; \mathrm{type}$; context extension yields a new context
$\Gamma.A$ extended with a fresh variable of type $A$,
along with a substitution $\pi : \Gamma.A \to A$ that forgets this fresh
variable.

We also have a notion of substitution extension: Given any substitution
$\sigma : \Gamma \to \Delta$, types $\Gamma \vdash A\; \mathrm{type}$ and
$\Delta \vdash B\; \mathrm{type}$, and term
$\Gamma.A \vdash t : B[\sigma]$, there exists some substitution
$\sigma. t : \Gamma.A \to \Delta.B$ such that the following square commutes.

~~~{.quiver}
\begin{tikzcd}
  {\Gamma.A} && {\Delta.B} \\
  \\
  \Gamma && \Delta
  \arrow["\sigma", from=3-1, to=3-3]
  \arrow["\pi"', from=1-1, to=3-1]
  \arrow["{\sigma.t}", from=1-1, to=1-3]
  \arrow["\pi"', from=1-3, to=3-3]
\end{tikzcd}
~~~

Furthermore, when the term $t$ is simply a variable from $\Gamma$, this
square is a [[pullback]] square!

Now that we've got a general idea of how context extension ought to
behave, we can begin to categorify. Our first step is to replace the
category of contexts with an arbitrary category $\cB$, and the types
with some [[fibration|fibred category]] $\cE$. We can then encode context extension
via a functor from $\cE$ to the [[codomain fibration]]. This is a somewhat
opaque definition, so it's worth unfolding somewhat. Note that the action
of such a functor on objects takes some object $x$ over $\Gamma$ in $\cE$
to a pair of an object we will suggestively name $\Gamma.A$ in $\cB$,
along with a map $\Gamma.A \to \Gamma$. Thus, the action on objects
yields both the extended context *and* the projection substitution. If we
inspect the action on morphisms of $\cE$, we see that it takes some
map $t : X \to_{\sigma} Y$ over $\sigma$ to a map $\sigma.t$ in $\cB$,
such that the following square commutes:

~~~{.quiver}
\begin{tikzcd}
  {\Gamma.A} && {\Delta.B} \\
  \\
  \Gamma && \Delta
  \arrow["\sigma", from=3-1, to=3-3]
  \arrow["\pi"', from=1-1, to=3-1]
  \arrow["{\sigma.t}", from=1-1, to=1-3]
  \arrow["\pi"', from=1-3, to=3-3]
\end{tikzcd}
~~~

Note that this is the exact same square as above!

Furthermore, this functor ought to be [fibred]; this captures the
situation where extending a substitution with a variable yields a
pullback square.

[fibred]: Cat.Displayed.Functor.html

We call such a [[fibred functor]] a **comprehension structure** on $\cE$[^1].

[^1]: Other sources call such fibrations **comprehension categories**,
but this is a bit of a misnomer, as they are structures on fibrations!

```agda
Comprehension : Type _
Comprehension = Vertical-fibred-functor E (Slices B)
```

Now, let's make all that earlier intuition formal. Let $\cE$ be a fibration,
and $P$ be a comprehension structure on $\cE$. We begin by defining
context extensions, along with their associated projections.


```agda
module Comprehension (fib : Cartesian-fibration E) (P : Comprehension) where opaque
  open Vertical-fibred-functor P
  open Cartesian-fibration fib

  _⨾_ : ∀ Γ → Ob[ Γ ] → Ob
  Γ ⨾ x = F₀' x .domain

  infixl 5 _⨾_

  _[_] : ∀ {Γ Δ} → Ob[ Δ ] → Hom Γ Δ → Ob[ Γ ]
  x [ σ ] =  base-change E fib σ .F₀ x

  πᶜ : ∀ {Γ x} → Hom (Γ ⨾ x) Γ
  πᶜ {x = x} = F₀' x .map
```

As $\cE$ is a fibration, we can reindex along the projection to obtain a
notion of weakening.

```agda
  weaken : ∀ {Γ} → (x : Ob[ Γ ]) → Ob[ Γ ] → Ob[ Γ ⨾ x ]
  weaken x y = y [ πᶜ ]
```

Furthermore, if $y$ is an object over $\Gamma$, then we have a map over
`πᶜ` between from the weakened form of $y$ to $y$.


```agda
  πᶜ' : ∀ {Γ} {x y : Ob[ Γ ]} → Hom[ πᶜ ] (weaken x y) y
  πᶜ' = has-lift.lifting πᶜ _

  πᶜ'-cartesian : ∀ {Γ x y} → is-cartesian E πᶜ (πᶜ' {Γ} {x} {y})
  πᶜ'-cartesian = has-lift.cartesian πᶜ _
```

Next, we define extension of substitutions, and show that they commute
with projections.

```agda
  _⨾ˢ_ : ∀ {Γ Δ x y} (σ : Hom Γ Δ) → Hom[ σ ] x y → Hom (Γ ⨾ x) (Δ ⨾ y)
  σ ⨾ˢ f = F₁' f .to

  infixl 8 _⨾ˢ_

  sub-proj : ∀ {Γ Δ x y} {σ : Hom Γ Δ} → (f : Hom[ σ ] x y) → πᶜ ∘ (σ ⨾ˢ f) ≡ σ ∘ πᶜ
  sub-proj f = sym $ F₁' f .commute
```

Crucially, when $f$ is cartesian, then the above square is a pullback.

```agda
  sub-pullback
    : ∀ {Γ Δ x y} {σ : Hom Γ Δ} {f : Hom[ σ ] x y}
    → is-cartesian E σ f
    → is-pullback B πᶜ σ (σ ⨾ˢ f) πᶜ
  sub-pullback {f = f} cart = cartesian→pullback B (F-cartesian f cart)

  module sub-pullback
    {Γ Δ x y} {σ : Hom Γ Δ} {f : Hom[ σ ] x y}
    (cart : is-cartesian E σ f)
    = is-pullback (sub-pullback cart)
```

We also obtain a map over $\sigma.f$ between the weakenings of $x$ and
$y$, which also commutes with projections.

```agda
  _⨾ˢ'_
    : ∀ {Γ Δ x y} (σ : Hom Γ Δ) → (f : Hom[ σ ] x y)
    → Hom[ σ ⨾ˢ f ] (weaken x x) (weaken y y)
  σ ⨾ˢ' f = has-lift.universal' πᶜ _ (sub-proj f) (f ∘' πᶜ')

  infixl 5 _⨾ˢ'_

  sub-proj'
    : ∀ {Γ Δ x y} {σ : Hom Γ Δ} → (f : Hom[ σ ] x y)
    → πᶜ' ∘' (σ ⨾ˢ' f) ≡[ sub-proj f ] f ∘' πᶜ'
  sub-proj' f = has-lift.commutesp πᶜ _ (sub-proj _) (f ∘' πᶜ')
```

If we extend the identity substitution with the identity morphism, we
obtain the identity morphism.

```agda
  sub-id : ∀ {Γ x} → id {Γ} ⨾ˢ id' {Γ} {x} ≡ id
  sub-id = ap to F-id'

  sub-id' : ∀ {Γ x} → (id ⨾ˢ' id') ≡[ sub-id {Γ} {x} ] id'
  sub-id' = symP $ has-lift.uniquep πᶜ _ _ (symP sub-id) (sub-proj id') id' $
    idr' _ ∙[] symP (idl' _)
```

Furthermore, extending a substitution with a pair of composites is the
same as composing the two extensions.

```agda
  sub-∘
    : ∀ {Γ Δ Ψ x y z}
    → {σ : Hom Δ Ψ} {δ : Hom Γ Δ} {f : Hom[ σ ] y z} {g : Hom[ δ ] x y}
    → (σ ∘ δ) ⨾ˢ (f ∘' g) ≡ (σ ⨾ˢ f) ∘ (δ ⨾ˢ g)
  sub-∘ {σ = σ} {δ = δ} {f = f} {g = g} = ap to F-∘'

  sub-∘'
    : ∀ {Γ Δ Ψ x y z}
    → {σ : Hom Δ Ψ} {δ : Hom Γ Δ} {f : Hom[ σ ] y z} {g : Hom[ δ ] x y}
    → ((σ ∘ δ) ⨾ˢ' (f ∘' g)) ≡[ sub-∘ ] (σ ⨾ˢ' f) ∘' (δ ⨾ˢ' g)
  sub-∘' = symP $ has-lift.uniquep πᶜ _ _ (symP sub-∘) (sub-proj _) _ $
    pulll[] _ (sub-proj' _)
    ∙[] extendr[] _ (sub-proj' _)
```

We can also define the substitution $\Gamma.A \to \Gamma.A.A$ that
duplicates the variable $A$ via the following pullback square.

~~~{.quiver}
\begin{tikzcd}
  {\Gamma.A} \\
  & {\Gamma.A.A} && {\Gamma.A} \\
  \\
  & {\Gamma.A} && \Gamma
  \arrow["\pi", from=4-2, to=4-4]
  \arrow["\pi"', from=2-2, to=4-2]
  \arrow["\pi", from=2-2, to=2-4]
  \arrow["\pi"', from=2-4, to=4-4]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=2-2, to=4-4]
  \arrow["id", curve={height=-12pt}, from=1-1, to=2-4]
  \arrow["id", curve={height=12pt}, from=1-1, to=4-2]
  \arrow["{\exists! \delta}", dashed, from=1-1, to=2-2]
\end{tikzcd}
~~~

```agda
  δᶜ : ∀ {Γ x} → Hom (Γ ⨾ x) (Γ ⨾ x ⨾ weaken x x)
  δᶜ = sub-pullback.universal (has-lift.cartesian πᶜ _) {p₁' = id} {p₂' = id} refl
```

This lets us easily show that applying projection after duplication is
the identity.

```agda
  proj-dup : ∀ {Γ x} → πᶜ ∘ δᶜ {Γ} {x} ≡ id
  proj-dup = sub-pullback.p₁∘universal (has-lift.cartesian πᶜ _)

  extend-proj-dup : ∀ {Γ x} → (πᶜ ⨾ˢ πᶜ') ∘ δᶜ {Γ} {x} ≡ id
  extend-proj-dup = sub-pullback.p₂∘universal (has-lift.cartesian πᶜ _)
```

We also obtain a substitution upstairs from the weakening of $x$ to the iterated
weakening of $x$.

```agda
  δᶜ' : ∀ {Γ} {x : Ob[ Γ ]} → Hom[ δᶜ ] (weaken x x) (weaken (weaken x x) (weaken x x))
  δᶜ' = has-lift.universal' πᶜ (weaken _ _) proj-dup id'
```

We also obtain similar lemmas detailing how duplication upstairs interacts with
projection.

```agda
  proj-dup' : ∀ {Γ x} → πᶜ' ∘' δᶜ' {Γ} {x} ≡[ proj-dup ] id'
  proj-dup' = has-lift.commutesp πᶜ _ proj-dup _

  extend-proj-dup' : ∀ {Γ x} → (πᶜ ⨾ˢ' πᶜ') ∘' δᶜ' {Γ} {x} ≡[ extend-proj-dup ] id'
  extend-proj-dup' = has-lift.uniquep₂ πᶜ _ _ _ _ _ _
    (pulll[] _ (sub-proj' _) ∙[] cancelr[] _ proj-dup')
    (idr' _)
```

From this, we can conclude that `δᶜ'`{.Agda} is cartesian. The
factorisation of $h' : u \to \Gamma.x.x$ is given by $\pi \circ h'$. This
is universal, as `δᶜ'`{.Agda} is given by the universal factorisation of
$\pi$.

```agda
  δᶜ'-cartesian : ∀ {Γ x} → is-cartesian E (δᶜ {Γ} {x}) δᶜ'
  δᶜ'-cartesian {Γ = Γ} {x = x} = cart where
    open is-cartesian

    cart : is-cartesian E (δᶜ {Γ} {x}) δᶜ'
    cart .universal m h' = hom[ cancell proj-dup ] (πᶜ' ∘' h')
    cart .commutes m h' = cast[] $
      unwrapr _
      ∙[] has-lift.uniquep₂ πᶜ _ _ (ap₂ _∘_ refl (cancell proj-dup)) refl _ _
        (cancell[] _ proj-dup')
        refl
    cart .unique m' p =
      has-lift.uniquep₂ πᶜ _ refl refl _ _ _
        refl
        (unwrapr _
        ∙[] ap₂ _∘'_ refl (ap₂ _∘'_ refl (sym p))
        ∙[] λ i → πᶜ' ∘' cancell[] _  proj-dup' {f' = m'} i)
```

We can also characterize how duplication interacts with extension.

```agda
  dup-extend
    : ∀ {Γ Δ x y} {σ : Hom Γ Δ} {f : Hom[ σ ] x y}
    → δᶜ ∘ (σ ⨾ˢ f) ≡ (σ ⨾ˢ f ⨾ˢ (σ ⨾ˢ' f)) ∘ δᶜ
  dup-extend {σ = σ} {f = f} =
    sub-pullback.unique₂ (has-lift.cartesian πᶜ _)
      {p = refl}
      (cancell proj-dup )
      (cancell extend-proj-dup)
      (pulll (sub-proj _)
       ∙ cancelr proj-dup)
      (pulll (sym sub-∘ ∙ ap₂ _⨾ˢ_ (sub-proj _) (sub-proj' _) ∙ sub-∘)
       ∙ cancelr extend-proj-dup)

  dup-extend'
    : ∀ {Γ Δ x y} {σ : Hom Γ Δ} {f : Hom[ σ ] x y}
    → δᶜ' ∘' (σ ⨾ˢ' f) ≡[ dup-extend ] (σ ⨾ˢ f ⨾ˢ' (σ ⨾ˢ' f)) ∘' δᶜ'
  dup-extend' {σ = σ} {f = f} =
    has-lift.uniquep₂ πᶜ _ _ _ _ _ _
      (cancell[] _ proj-dup')
      (pulll[] _ (sub-proj' (σ ⨾ˢ' f)) ∙[] cancelr[] _ proj-dup')
```

```agda
  extend-dup² : ∀ {Γ x} → (δᶜ {Γ} {x} ⨾ˢ δᶜ') ∘ δᶜ ≡ δᶜ ∘ δᶜ
  extend-dup² =
    sub-pullback.unique₂ (has-lift.cartesian πᶜ _)
      {p = refl}
      (pulll (sub-proj _) ∙ cancelr proj-dup)
      (cancell (sym sub-∘ ∙ ap₂ _⨾ˢ_ proj-dup proj-dup' ∙ sub-id))
      (cancell proj-dup)
      (cancell extend-proj-dup)

  extend-dup²' : ∀ {Γ x} → (δᶜ {Γ} {x} ⨾ˢ' δᶜ') ∘' δᶜ' ≡[ extend-dup² ] δᶜ' ∘' δᶜ'
  extend-dup²' = has-lift.uniquep₂ πᶜ
    _ _ _ _ _ _
    (pulll[] _ (sub-proj' δᶜ') ∙[] cancelr[] _ proj-dup')
    (cancell[] _ proj-dup')
```

Note that we can extend the operation of context extension to a functor
from the [[total category]] of $\cE$ to $\cB$, that takes every pair
$(\Gamma, A)$ to $\Gamma.A$

```agda
  Extend : Functor (∫ E) B
  Extend .F₀ (Γ , x) = Γ ⨾ x
  Extend .F₁ (total-hom σ f) = σ ⨾ˢ f
  Extend .F-id = ap to F-id'
  Extend .F-∘ f g = ap to F-∘'
```

There is also a natural transformation from this functor into the
projection out of the total category of $\cE$, where each component
of is a projection $\Gamma.A \to \Gamma$.

```agda
  proj : Extend => πᶠ E
  proj .η (Γ , x) = πᶜ
  proj .is-natural (Γ , x) (Δ , y) (total-hom σ f) =
    sub-proj f
```

## Comprehension structures as comonads

Comprehension structures on fibrations $\cE$ induce [comonads] on the
[[total category]] of $\cE$. These comonads are particularly nice: all
of the counits are [[cartesian morphisms]], and every square of the
following form is a pullback square, provided that $f$ is cartesian.

[comonads]: Cat.Diagram.Comonad.html

~~~{.quiver}
\begin{tikzcd}
  {W (\Gamma, X)} && {W (\Delta, Y)} \\
  \\
  {(\Gamma, X)} && {(\Delta, Y)}
  \arrow["{(\sigma, f)}"', from=3-1, to=3-3]
  \arrow["\eps"', from=1-1, to=3-1]
  \arrow["\eps", from=1-3, to=3-3]
  \arrow["{W (\sigma, f)}", from=1-1, to=1-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-1, to=3-3]
\end{tikzcd}
~~~

We call such comonads **comprehension comonads**.

```agda
record Comprehension-comonad : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
  no-eta-equality
  field
    comprehend : Functor (∫ E) (∫ E)
    comonad    : Comonad-on comprehend

  open Comonad-on comonad public

  field
    counit-cartesian
      : ∀ {Γ x} → is-cartesian E (counit.ε (Γ , x) .hom) (counit.ε (Γ , x) .preserves)
    cartesian-pullback
      : (∀ {Γ Δ x y} {σ : Hom Γ Δ} {f : Hom[ σ ] x y}
      → is-cartesian E σ f
      → is-pullback (∫ E)
          (counit.ε (Γ , x)) (total-hom σ f)
          (W₁ (total-hom σ f)) (counit.ε (Δ , y)))
```

As promised, comprehension structures on $\cE$ yield comprehension
comonads.

```agda
Comprehension→comonad
  : Cartesian-fibration E
  → Comprehension
  → Comprehension-comonad
Comprehension→comonad fib P = comp-comonad where
  open Comprehension fib P
  open Comonad-on
```

We begin by constructing the endofunctor $\int E \to \int E$, which maps
a pair $\Gamma, X$ to the extension $\Gamma.X$, along with the weakening
of $X$.

```agda
  comprehend : Functor (∫ E) (∫ E)
  comprehend .F₀ (Γ , x) = Γ ⨾ x , weaken x x
  comprehend .F₁ (total-hom σ f) = total-hom (σ ⨾ˢ f) (σ ⨾ˢ' f)
  comprehend .F-id = total-hom-path E sub-id sub-id'
  comprehend .F-∘ (total-hom σ f) (total-hom δ g) =
    total-hom-path E sub-∘ sub-∘'
```

The counit is given by the projection substitution, and comultiplication
is given by duplication.

```agda
  comonad : Comonad-on comprehend
  comonad .counit .η (Γ , x) =
    total-hom πᶜ πᶜ'
  comonad .counit .is-natural (Γ , x) (Δ , g) (total-hom σ f) =
    total-hom-path E (sub-proj f) (sub-proj' f)
  comonad .comult .η (Γ , x) =
    total-hom δᶜ δᶜ'
  comonad .comult .is-natural (Γ , x) (Δ , g) (total-hom σ f) =
    total-hom-path E dup-extend dup-extend'
  comonad .δ-idl =
    total-hom-path E extend-proj-dup extend-proj-dup'
  comonad .δ-idr =
    total-hom-path E proj-dup proj-dup'
  comonad .δ-assoc =
    total-hom-path E extend-dup² extend-dup²'
```

To see that this comonad is a comprehension comonad, note that the
projection substitution is cartesian. Furthermore, we can construct
a pullback square in the total category of $\cE$ from one in the base,
provided that two opposing sides are cartesian, which the projection
morphism most certainly is!

```agda
  comp-comonad : Comprehension-comonad
  comp-comonad .Comprehension-comonad.comprehend = comprehend
  comp-comonad .Comprehension-comonad.comonad = comonad
  comp-comonad .Comprehension-comonad.counit-cartesian = πᶜ'-cartesian
  comp-comonad .Comprehension-comonad.cartesian-pullback cart =
    cartesian+pullback→total-pullback E
      πᶜ'-cartesian πᶜ'-cartesian
      (sub-pullback cart)
      (cast[] (symP (sub-proj' _)))
```

We also show that comprehension comonads yield comprehension structures.

```agda
Comonad→comprehension
  : Cartesian-fibration E
  → Comprehension-comonad
  → Comprehension
```

We begin by constructing a [vertical functor] $\cE \to B^{\to}$ that maps
an $x$ lying over $\Gamma$ to the base component of the counit
$\eps : W(\Gamma, X) \to (\Gamma, X)$.

[vertical functor]: Cat.Displayed.Functor.html

```agda
Comonad→comprehension fib comp-comonad = comprehension where
  open Comprehension-comonad comp-comonad
  open Vertical-functor
  open is-pullback

  vert : Vertical-functor E (Slices B)
  vert .F₀' {Γ} x = cut (counit.ε (Γ , x) .hom)
  vert .F₁' {f = σ} f =
    slice-hom (W₁ (total-hom σ f) .hom)
      (sym (ap hom (counit.is-natural _ _ _)))
  vert .F-id' = Slice-path B (ap hom W-id)
  vert .F-∘' = Slice-path B (ap hom (W-∘ _ _))
```

To see that this functor is fibred, recall that pullbacks in the
codomain fibration are given by pullbacks. Furthermore, if we
have a pullback square in the total category of $\cE$ where two
opposing sides are cartesian, then we have a corresponding pullback
square in $\cB$. As the comonad is a comprehension comonad, counit is
cartesian, which finishes off the proof.

```agda
  fibred : is-vertical-fibred vert
  fibred {f = σ} f cart =
    pullback→cartesian B $
    cartesian+total-pullback→pullback E fib
      counit-cartesian counit-cartesian
      (cartesian-pullback cart)

  comprehension : Comprehension
  comprehension .Vertical-fibred-functor.vert = vert
  comprehension .Vertical-fibred-functor.F-cartesian = fibred
```
