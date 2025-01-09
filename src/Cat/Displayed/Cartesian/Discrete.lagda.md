---
description: |
  We define discrete fibrations,
  and explore their relations to presheaves.
---
<!--
```agda
open import Cat.Displayed.Instances.Elements
open import Cat.Displayed.Cartesian
open import Cat.Displayed.Functor
open import Cat.Instances.Functor
open import Cat.Displayed.Fibre
open import Cat.Displayed.Base
open import Cat.Displayed.Path
open import Cat.Prelude

import Cat.Displayed.Morphism
```
-->

```agda
module Cat.Displayed.Cartesian.Discrete where
```

<!--
```agda
open Cartesian-fibration
open Cartesian-lift
open is-cartesian
```
-->

# Discrete fibrations

A **discrete fibration** is a [[displayed category]] whose [[fibre
categories]] are all _discrete categories_: thin, univalent groupoids.
Since thin, univalent groupoids are sets, a discrete fibration over
$\cB$ is an alternate way of encoding a presheaf over $\cB$, i.e., a
functor $\cB\op\to\Sets$. Here, we identify a purely fibrational
property that picks out the discrete fibrations among the displayed
categories, without talking about the fibres directly.

A discrete fibration is a displayed category such that each type of
displayed objects is a set, and such that, for each right corner

~~~{.quiver}
\[\begin{tikzcd}
  & {y'} \\
  x & {y\text{,}}
  \arrow[lies over, from=1-2, to=2-2]
  \arrow["f"', from=2-1, to=2-2]
\end{tikzcd}\]
~~~

there is a contractible space of objects $x'$ over $x$ equipped with
maps $x' \to_f y'$.

<!--

```agda
module _ {o ℓ o' ℓ'} {B : Precategory o ℓ} (E : Displayed B o' ℓ') where
  private
    module B = Precategory B
    module E = Displayed E
    open Cat.Displayed.Morphism E
    open Displayed E
```

-->

```agda
  record Discrete-fibration : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    field
      fibre-set : ∀ x → is-set E.Ob[ x ]
      lifts
        : ∀ {x y} (f : B.Hom x y) (y' : E.Ob[ y ])
        → is-contr (Σ[ x' ∈ E.Ob[ x ] ] E.Hom[ f ] x' y')
```

## Discrete fibrations are cartesian

To prove that discrete fibrations deserve the name discrete
_fibrations_, we prove that any discrete fibration is a [[Cartesian
fibration]]. By assumption, every right corner has a unique lift, which
is in particular a lift: we just have to show that the lift is
Cartesian.

```agda
  discrete→cartesian : Discrete-fibration → Cartesian-fibration E
  discrete→cartesian disc = r where
    open Discrete-fibration disc
    r : Cartesian-fibration E
    r .has-lift f y' .x' = lifts f y' .centre .fst
    r .has-lift f y' .lifting = lifts f y' .centre .snd
```

So suppose we have an open diagram

~~~{.quiver}
\[\begin{tikzcd}
  {u'} \\
  & {a'} && {b'} \\
  u \\
  & a && {b,}
  \arrow["{f'}"', from=2-2, to=2-4]
  \arrow["f", from=4-2, to=4-4]
  \arrow[lies over, from=2-2, to=4-2]
  \arrow[lies over, from=2-4, to=4-4]
  \arrow["m"', from=3-1, to=4-2]
  \arrow[lies over, from=1-1, to=3-1]
  \arrow["{h'}", curve={height=-12pt}, from=1-1, to=2-4]
\end{tikzcd}\]
~~~

where $f' : a' \to b'$ is the unique lift of $f$ along $b'$. We need to
find a map $u' \to_m a'$. Observe that we have a right corner (with
vertices $u$ and $a'$ over $a$), so that we an object $u_2$ over $u$ and
map $l : u_2 \to_m a'$. Initially, this looks like it might not help,
but observe that $u' \xto{h'}_{f \circ m} b'$ and $u_2 \xto{l}_{u} a'
\xto{f'}_f b'$ are lifts of the right corner with base given by $u \to a
\to b$, so that by uniqueness, $u2 = u'$: thus, we can use $l$ as our
map $u' \to a'$.

```agda
    r .has-lift f y' .cartesian .universal {u} {u'} m h' =
      subst (λ x → E.Hom[ m ] x (lifts f y' .centre .fst))
        (ap fst $ is-contr→is-prop (lifts (f B.∘ m) y')
          (_ , lifts f y' .centre .snd E.∘' lifts m _ .centre .snd)
          (u' , h'))
        (lifts m (lifts f y' .centre .fst) .centre .snd)
    r .has-lift f y' .cartesian .commutes m h' =
      Σ-inj-set (fibre-set _) $ is-contr→is-prop (lifts (f B.∘ m) y') _ _
    r .has-lift f y' .cartesian .unique {u} {u'} {m} m' x =
      Σ-inj-set (fibre-set u) $ is-contr→is-prop (lifts m _) (u' , m') (u' , _)
```

## Fibres of discrete fibrations

Let $x$ be an object of $\cB$. Let us ponder the fibre $\cE^*(x)$:
we know that it is strict, since by assumption there is a _set_ of
objects over $x$. Let us show also that it is thin: imagine that we have
two parallel, vertical arrows $f, g : a \to_{\id} b$. These assemble
into a diagram like

~~~{.quiver}
\[\begin{tikzcd}
  {a'} && {b'} && {a'} \\
  \\
  x && x && {x\text{,}}
  \arrow["f", from=1-1, to=1-3]
  \arrow["g"', from=1-5, to=1-3]
  \arrow[lies over, from=1-1, to=3-1]
  \arrow[lies over, from=1-3, to=3-3]
  \arrow[lies over, from=1-5, to=3-5]
  \arrow["{\mathrm{id}}"{description}, from=3-1, to=3-3]
  \arrow["{\mathrm{id}}"{description}, from=3-5, to=3-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-1, to=3-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125, rotate=-90}, draw=none, from=1-5, to=3-3]
\end{tikzcd}\]
~~~

whence we see that $(a', f)$ and $(a', g)$ are both lifts for the lower
corner given by lifting the identity map along $b'$ --- so, since lifts
are unique, we have $f = g$.

```agda
  discrete→thin-fibres
    : ∀ x → Discrete-fibration → ∀ {a b} → is-prop (Fibre E x .Precategory.Hom a b)
  discrete→thin-fibres x disc {a} {b} f g =
    Σ-inj-set (fibre-set x) $
      is-contr→is-prop (lifts B.id b) (a , f) (a , g)
    where open Discrete-fibration disc
```

## Morphisms in discrete fibrations

If $\cE$ is a discrete fibration, then the only vertical morphisms
are identities. This follows directly from lifts being contractible.

```agda
  discrete→vertical-id
    : Discrete-fibration
    → ∀ {x} {x'' : E.Ob[ x ]} (f' : Σ[ x' ∈ E.Ob[ x ] ] (E.Hom[ B.id ] x' x''))
    → (x'' , E.id') ≡ f'
  discrete→vertical-id disc {x'' = x''} f' =
    sym (lifts B.id _ .paths (x'' , E.id')) ∙ lifts B.id x'' .paths f'
    where open Discrete-fibration disc
```

We can use this fact in conjunction with the fact that all fibres are thin to show
that every vertical morphism in a discrete fibration is invertible.

```agda
  discrete→vertical-invertible
    : Discrete-fibration
    → ∀ {x} {x' x'' : E.Ob[ x ]} → (f' : E.Hom[ B.id ] x' x'') → is-invertible↓ f'
  discrete→vertical-invertible disc {x' = x'} {x'' = x''} f' =
    make-invertible↓
      (subst (λ x' → E.Hom[ B.id ] x'' x') x''≡x' E.id')
      (to-pathp (discrete→thin-fibres _ disc _ _))
      (to-pathp (discrete→thin-fibres _ disc _ _))
    where
      x''≡x' : x'' ≡ x'
      x''≡x' = ap fst (discrete→vertical-id disc (x' , f'))
```

## Discrete fibrations are presheaves {defines="discrete-fibrations-are-presheaves"}

As noted earlier, a discrete fibration over $\cB$ encodes the same
data as a presheaf on $\cB$. First, let us show that we can construct
a presheaf from a discrete fibration.

<!--
```agda
module _ {o ℓ} (B : Precategory o ℓ)  where
  private
    module B = Precategory B
```
-->

```agda
  discrete→presheaf : ∀ {o' ℓ'} (E : Displayed B o' ℓ') → Discrete-fibration E
                    → Functor (B ^op) (Sets o')
  discrete→presheaf {o' = o'} E disc = psh where
    module E = Displayed E
    open Discrete-fibration disc
```

For each object in $X : \cB$, we take the set of objects $E$ that
lie over $X$. The functorial action of `f : Hom X Y` is then constructed
by taking the domain of the lift of `f`. Functoriality follows by
uniqueness of the lifts.

```agda
    psh : Functor (B ^op) (Sets o')
    psh .Functor.F₀ X = el E.Ob[ X ] (fibre-set X)
    psh .Functor.F₁ f X' = lifts f X' .centre .fst
    psh .Functor.F-id = funext λ X' → ap fst (lifts B.id X' .paths (X' , E.id'))
    psh .Functor.F-∘ {X} {Y} {Z} f g = funext λ X' →
      let Y' : E.Ob[ Y ]
          Y' = lifts g X' .centre .fst

          g' : E.Hom[ g ] Y' X'
          g' = lifts g X' .centre .snd

          Z' : E.Ob[ Z ]
          Z' = lifts f Y' .centre .fst

          f' : E.Hom[ f ] Z' Y'
          f' = lifts f Y' .centre .snd
      in ap fst (lifts (g B.∘ f) X' .paths (Z' , (g' E.∘' f' )))
```

To construct a discrete fibration from a presheaf $P$, we take the
[(displayed) category of elements] of $P$. This is a natural choice,
as it encodes the same data as $P$, just rendered down into a soup
of points and bits of functions. Discreteness follows immediately
from the contractibilty of singletons.

[(displayed) category of elements]: Cat.Displayed.Instances.Elements.html

```agda
  presheaf→discrete : ∀ {κ} → Functor (B ^op) (Sets κ)
                    → Σ[ E ∈ Displayed B κ κ ] Discrete-fibration E
  presheaf→discrete {κ = κ} P = ∫ B P , discrete where
    module P = Functor P

    discrete : Discrete-fibration (∫ B P)
    discrete .Discrete-fibration.fibre-set X =
      P.₀ X .is-tr
    discrete .Discrete-fibration.lifts f P[Y] =
      contr (P.₁ f P[Y] , refl) Singleton-is-contr
```

We conclude by proving that the two maps defined above are, in fact,
inverses. Most of the complexity is in [characterising paths between
displayed categories][disppath], but that doesn't mean that the proof
here is entirely trivial, either. Well, first, we note that one
direction _is_ trivial: modulo differences in the proofs of
functoriality, which do not matter for identity, turning a functor into
a discrete fibration and back is the identity.

[disppath]: Cat.Displayed.Path.html

```agda
  open is-iso

  presheaf≃discrete : ∀ {κ} → is-iso (presheaf→discrete {κ = κ})
  presheaf≃discrete .inv  (d , f) = discrete→presheaf d f
  presheaf≃discrete .linv x       = Functor-path (λ _ → n-path refl) λ _ → refl
```

The other direction is where the complication lies. Given a discrete
fibration $P \liesover X$, how do we show that $\int P \equiv P$? Well,
by the aforementioned characterisation of paths in displayed categories,
it'll suffice to construct a functor $(\int P) \to P$ (lying over the
identity), then show that this functor has an invertible action on
objects, and an invertible action on morphisms.

```agda
  presheaf≃discrete .rinv (P , p-disc) = Σ-prop-path hl ∫≡dx where
    open Discrete-fibration p-disc
    open Displayed-functor
    open Displayed P
```

The functor will send an object $c \liesover x$ to that same object $c$;
This is readily seen to be invertible. But the action on morphisms is
slightly more complicated. Recall that, since $P$ is a discrete
fibration, every span $b' \liesover b \xot{f} a$ has a contractible
space of Cartesian lifts $(a', f')$. Our functor must, given objects
$a'', b'$, a map $f : a \to b$, and a proof that $a'' = a'$, produce a
map $a'' \to_f b$ --- so we can take the canonical $f' : a' \to_f b$ and
transport it over the given $a'' = a'$.

```agda
    pieces : Displayed-functor (∫ B (discrete→presheaf P p-disc)) P Id
    pieces .F₀' x = x
    pieces .F₁' {f = f} {a'} {b'} x =
      subst (λ e → Hom[ f ] e b') x $ lifts f b' .centre .snd
```

This transport _threatens_ to throw a spanner in the works, since it is
an equation between objects (over $a$). But since $P$ is a discrete
fibration, the space of objects over $a$ is a set, so this equation
_can't_ ruin our day. Directly from the uniqueness of $(a', f')$ we
conclude that we've put together a functor.

```agda
    pieces .F-id' = from-pathp (ap snd (lifts _ _ .paths _))
    pieces .F-∘' {f = f} {g} {a'} {b'} {c'} {f'} {g'} =
      ap (λ e → subst (λ e → Hom[ f B.∘ g ] e c') e
            (lifts _ _ .centre .snd)) (fibre-set _ _ _ _ _)
      ∙ from-pathp (ap snd (lifts _ _ .paths _))
```

It remains to show that, given a map $a'' \to b$ (and the rest of the
data $a$, $b$, $f : a \to b$, $b' \liesover b$), we can recover a proof
that $a''$ is the chosen lift $a'$. But again, lifts are unique, so this
is immediate.

```agda
    ∫≡dx : ∫ B (discrete→presheaf P p-disc) ≡ P
    ∫≡dx = Displayed-path pieces (λ _ → id-equiv) (is-iso→is-equiv p) where
      p : ∀ {a b} {f : B.Hom a b} {a'} {b'} → is-iso (pieces .F₁' {f = f} {a'} {b'})
      p .inv f  = ap fst $ lifts _ _ .paths (_ , f)
      p .rinv p = from-pathp (ap snd (lifts _ _ .paths _))
      p .linv p = fibre-set _ _ _ _ _
```

We must additionally show that the witness that $P$ is a discrete
fibration will survive a round-trip through the type of presheaves, but
this witness lives in a proposition (it is a pair of propositions), so
it survives automatically.

```agda
    private unquoteDecl eqv = declare-record-iso eqv (quote Discrete-fibration)
    hl : ∀ x → is-prop _
    hl x = Iso→is-hlevel! 1 eqv
```
