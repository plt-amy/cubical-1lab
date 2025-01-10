<!--
```agda
open import Cat.Prelude

import Cat.Reasoning
```
-->

```agda
module Cat.Diagram.Pullback where
```

# Pullbacks {defines=pullback}

<!--
```agda
module _ {o ℓ} (C : Precategory o ℓ) where
  open Cat.Reasoning C
  private variable
    P' X Y Z : Ob
    h p₁' p₂' : Hom X Y
```
-->

A **pullback** $X \times_Z Y$ of $f : X \to Z$ and $g : Y \to Z$ is the
[[product]] of $f$ and $g$ in the category $\cC/Z$, the category of
objects fibred over $Z$. We note that the fibre of $X \times_Z Y$ over
some element $x$ of $Z$ is the product of the fibres of $f$ and $g$ over
$x$; Hence the pullback is also called the **fibred product**.

```agda
  record is-pullback {P} (p₁ : Hom P X) (f : Hom X Z) (p₂ : Hom P Y) (g : Hom Y Z)
    : Type (o ⊔ ℓ) where

    no-eta-equality
    field
      square   : f ∘ p₁ ≡ g ∘ p₂
```

The concrete incarnation of the abstract nonsense above is that a
pullback turns out to be a universal square like the one below. Since it
is a product, it comes equipped with projections $\pi_1$ and $\pi_2$
onto its factors; Since isn't merely a product of $X$ and $Y$, but
rather of $X$ and $Y$ considered as objects over $Z$ in a specified way,
overall square has to commute.

~~~{.quiver}
\[\begin{tikzcd}
  P && Y \\
  \\
  X && Z
  \arrow["g", from=1-3, to=3-3]
  \arrow["f"', from=3-1, to=3-3]
  \arrow["{\pi_1}"', from=1-1, to=3-1]
  \arrow["{\pi_2}", from=1-1, to=1-3]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=1-1, to=3-3]
\end{tikzcd}\]
~~~

```agda
      universal : ∀ {P'} {p₁' : Hom P' X} {p₂' : Hom P' Y}
               → f ∘ p₁' ≡ g ∘ p₂' → Hom P' P
      p₁∘universal : {p : f ∘ p₁' ≡ g ∘ p₂'} → p₁ ∘ universal p ≡ p₁'
      p₂∘universal : {p : f ∘ p₁' ≡ g ∘ p₂'} → p₂ ∘ universal p ≡ p₂'

      unique : {p : f ∘ p₁' ≡ g ∘ p₂'} {lim' : Hom P' P}
             → p₁ ∘ lim' ≡ p₁'
             → p₂ ∘ lim' ≡ p₂'
             → lim' ≡ universal p

    unique₂
      : {p : f ∘ p₁' ≡ g ∘ p₂'} {lim' lim'' : Hom P' P}
      → p₁ ∘ lim' ≡ p₁' → p₂ ∘ lim' ≡ p₂'
      → p₁ ∘ lim'' ≡ p₁' → p₂ ∘ lim'' ≡ p₂'
      → lim' ≡ lim''
    unique₂ {p = o} p q r s = unique {p = o} p q ∙ sym (unique r s)
```

<!--
```agda
    pullback-univ
      : ∀ {O} → Hom O P ≃ (Σ (Hom O X) λ h → Σ (Hom O Y) λ h' → f ∘ h ≡ g ∘ h')
    pullback-univ .fst h = p₁ ∘ h , p₂ ∘ h , extendl square
    pullback-univ .snd = is-iso→is-equiv λ where
      .is-iso.inv (f , g , α) → universal α
      .is-iso.rinv x → Σ-pathp p₁∘universal $ Σ-prop-pathp (λ _ _ → hlevel 1) p₂∘universal
      .is-iso.linv x → sym (unique refl refl)
```
-->

By universal, we mean that any other "square" (here the second "square"
has corners $P'$, $X$, $Y$, $Z$ --- it's a bit bent) admits a unique
factorisation that passes through $P$; We can draw the whole situation
as in the diagram below. Note the little corner on $P$, indicating that
the square is a pullback.

~~~{.quiver}
\[\begin{tikzcd}
  {P'} \\
  & P && Y \\
  \\
  & X && Z
  \arrow["g", from=2-4, to=4-4]
  \arrow["f"', from=4-2, to=4-4]
  \arrow["{\pi_1}"', from=2-2, to=4-2]
  \arrow["{\pi_2}", from=2-2, to=2-4]
  \arrow["{\exists! }", dashed, from=1-1, to=2-2]
  \arrow["{p_2}", curve={height=-12pt}, from=1-1, to=2-4]
  \arrow["{p_1}", curve={height=12pt}, from=1-1, to=4-2]
  \arrow["\lrcorner"{anchor=center, pos=0.125}, draw=none, from=2-2, to=4-4]
\end{tikzcd}\]
~~~

We provide a convenient packaging of the pullback and the projection
maps:

```agda
  record Pullback {X Y Z} (f : Hom X Z) (g : Hom Y Z) : Type (o ⊔ ℓ) where
    no-eta-equality
    field
      {apex} : Ob
      p₁ : Hom apex X
      p₂ : Hom apex Y
      has-is-pb : is-pullback p₁ f p₂ g

    open is-pullback has-is-pb public
```

<!--
```agda
module _ {o ℓ} {C : Precategory o ℓ} where
  open Cat.Reasoning C
  private variable
    P' X Y Z : Ob
    h p₁' p₂' : Hom X Y

  is-pullback-is-prop : ∀ {P} {p₁ : Hom P X} {f : Hom X Z} {p₂ : Hom P Y} {g : Hom Y Z} → is-prop (is-pullback C p₁ f p₂ g)
  is-pullback-is-prop {X = X} {Y = Y} {p₁ = p₁} {f} {p₂} {g} x y = q where
    open is-pullback
    p : Path (∀ {P'} {p₁' : Hom P' X} {p₂' : Hom P' Y} → f ∘ p₁' ≡ g ∘ p₂' → _) (x .universal) (y .universal)
    p i sq = y .unique {p = sq} (x .p₁∘universal {p = sq}) (x .p₂∘universal) i
    q : x ≡ y
    q i .square = Hom-set _ _ _ _ (x .square) (y .square) i
    q i .universal = p i
    q i .p₁∘universal {p₁' = p₁'} {p = sq} = is-prop→pathp (λ i → Hom-set _ _ (p₁ ∘ p i sq) p₁') (x .p₁∘universal) (y .p₁∘universal) i
    q i .p₂∘universal {p = sq} = is-prop→pathp (λ i → Hom-set _ _ (p₂ ∘ p i sq) _) (x .p₂∘universal) (y .p₂∘universal) i
    q i .unique {p = sq} {lim' = lim'} c₁ c₂ = is-prop→pathp (λ i → Hom-set _ _ lim' (p i sq)) (x .unique c₁ c₂) (y .unique c₁ c₂) i

  instance
    H-Level-is-pullback : ∀ {P} {p₁ : Hom P X} {f : Hom X Z} {p₂ : Hom P Y} {g : Hom Y Z} {n} → H-Level (is-pullback C p₁ f p₂ g) (suc n)
    H-Level-is-pullback = prop-instance is-pullback-is-prop

  subst-is-pullback
    : ∀ {P} {p₁ p₁' : Hom P X} {f f' : Hom X Z} {p₂ p₂' : Hom P Y} {g g' : Hom Y Z}
    → p₁ ≡ p₁' → f ≡ f' → p₂ ≡ p₂' → g ≡ g'
    → is-pullback C p₁ f p₂ g
    → is-pullback C p₁' f' p₂' g'
  subst-is-pullback p q r s = transport (λ i → is-pullback C (p i) (q i) (r i) (s i))
```
-->

## Kernel pairs {defines="kernel-pair"}

The **kernel pair** of a morphism $f : X \to Y$ (if it exists) is
the pullback of $f$ along itself. Intuitively, one should think
of a kernel pair as a partition of $X$ induced by the preimage of
$f$.

<!--
```agda
module _ {o ℓ} (C : Precategory o ℓ) where
  open Cat.Reasoning C
```
-->

```agda
  is-kernel-pair : ∀ {P X Y} → Hom P X → Hom P X → Hom X Y → Type _
  is-kernel-pair p1 p2 f = is-pullback C p1 f p2 f
```

<!--
```agda
module _ {o ℓ} {C : Precategory o ℓ} where
  open Cat.Reasoning C
  private variable
    X Y P : Ob
```
-->

Note that each of the projections out of the kernel pair of $f$ are
[[epimorphisms]]. Without loss of generality, we will focus our
attention on the first projection.


```agda
  is-kernel-pair→epil
    : ∀ {p1 p2 : Hom P X} {f : Hom X Y}
    → is-kernel-pair C p1 p2 f
    → is-epic p1
```

Recall that a morphism is epic if it has a [[section]]; that is,
some morphism $u$ such that $p_1 \circ u = \id$. We can construct
such a $u$ by applying the universal property of the pullback to
the following diagram.

~~~{.quiver}
\begin{tikzcd}
  X \\
  & P && X \\
  \\
  & X && Y
  \arrow["u", from=1-1, to=2-2]
  \arrow["id", curve={height=-12pt}, from=1-1, to=2-4]
  \arrow["id"', curve={height=12pt}, from=1-1, to=4-2]
  \arrow["{p_2}", from=2-2, to=2-4]
  \arrow["{p_1}"', from=2-2, to=4-2]
  \arrow["f", from=2-4, to=4-4]
  \arrow["f"', from=4-2, to=4-4]
\end{tikzcd}
~~~

```agda
  is-kernel-pair→epil {p1 = p1} is-kp =
    has-section→epic $
    make-section
      (universal refl)
      p₁∘universal
    where open is-pullback is-kp
```

<!--
```agda
  is-kernel-pair→epir
    : ∀ {P} {p1 p2 : Hom P X} {f : Hom X Y}
    → is-kernel-pair C p1 p2 f
    → is-epic p2
  is-kernel-pair→epir {p2 = p2} is-kp =
    has-section→epic $
    make-section
      (universal refl)
      p₂∘universal
    where open is-pullback is-kp
```
-->

If $f$ is a [[monomorphism]], then its kernel pair always exists, and
is given by $(\id, \id)$.

```agda
  monic→id-kernel-pair
    : ∀ {f : Hom X Y}
    → is-monic f
    → is-kernel-pair C id id f
```

Clearly, the square $f \circ \id = f \circ \id$ commutes, so the tricky
bit will be constructing a universal morphism. If $f \circ p_1 = f \circ
p_2$ for some $p_1, p_2 : P \to X$, then we can simply use one of $p_1$
or $p_2$ for our universal map; the choice we make does not matter, as
we can obtain $p_1 = p_2$ from the fact that $f$ is monic! The rest of
the universal property follows directly from this lovely little
observation.

```agda
  monic→id-kernel-pair {f = f} f-monic = id-kp where
    open is-pullback

    id-kp : is-kernel-pair C id id f
    id-kp .square = refl
    id-kp .universal {p₁' = p₁'} _ = p₁'
    id-kp .p₁∘universal = idl _
    id-kp .p₂∘universal {p = p} = idl _ ∙ f-monic _ _ p
    id-kp .unique p q = sym (idl _) ∙ p
```

Conversely, if $(\id, \id)$ is the kernel pair of $f$, then $f$ is
monic. Suppose that $f \circ g = f \circ h$ for some $g, h : A \to X$,
and note that both $g$ and $h$ are equal to the universal map obtained
via the square $f \circ g = f \circ h$.

```agda
  id-kernel-pair→monic
    : ∀ {f : Hom X Y}
    → is-kernel-pair C id id f
    → is-monic f
  id-kernel-pair→monic {f = f} id-kp g h p =
    g                ≡˘⟨ p₁∘universal ⟩
    id ∘ universal p ≡⟨ p₂∘universal ⟩
    h                ∎
    where open is-pullback id-kp
```

We can strengthen this result by noticing that if $(p, p)$ is the kernel
pair of $f$ for some $P : \cC, p : P \to X$, then $(\id, \id)$ is also a
kernel pair of $f$.

```agda
  same-kernel-pair→id-kernel-pair
    : ∀ {P} {p : Hom P X} {f : Hom X Y}
    → is-kernel-pair C p p f
    → is-kernel-pair C id id f
```

As usual, the difficulty is constructing the universal map. Suppose
that $f \circ p_1 = f \circ p_2$ for some $P' : \cC, p_1, p_2 : P' \to X$,
as in the following diagram:

~~~{.quiver}
\begin{tikzcd}
  {P'} \\
  & P && X \\
  \\
  & X && Y
  \arrow["{p_2}", curve={height=-12pt}, from=1-1, to=2-4]
  \arrow["{p_1}"', curve={height=12pt}, from=1-1, to=4-2]
  \arrow["p", from=2-2, to=2-4]
  \arrow["p"', from=2-2, to=4-2]
  \arrow["f", from=2-4, to=4-4]
  \arrow["f"', from=4-2, to=4-4]
\end{tikzcd}
~~~

This diagram is conspicuously missing a morphism, so let's fill
it in by using the universal property of the kernel pair.

~~~{.quiver}
\begin{tikzcd}
  {P'} \\
  & P && X \\
  \\
  & X && Y
  \arrow["u", dashed, from=1-1, to=2-2]
  \arrow["{p_2}", curve={height=-12pt}, from=1-1, to=2-4]
  \arrow["{p_1}"', curve={height=12pt}, from=1-1, to=4-2]
  \arrow["p", from=2-2, to=2-4]
  \arrow["p"', from=2-2, to=4-2]
  \arrow["f", from=2-4, to=4-4]
  \arrow["f"', from=4-2, to=4-4]
\end{tikzcd}
~~~

Next, note that $p \circ u$ factorizes both $p_1$ *and* $p_2$; moreover,
it is the unique such map!

```agda
  same-kernel-pair→id-kernel-pair {p = p} {f = f} p-kp = id-kp where
    open is-pullback

    id-kp : is-kernel-pair C id id f
    id-kp .square = refl
    id-kp .universal q = p ∘ p-kp .universal q
    id-kp .p₁∘universal {p = q} = idl _ ∙ p-kp .p₁∘universal
    id-kp .p₂∘universal {p = q} = idl _ ∙ p-kp .p₂∘universal
    id-kp .unique q r = (sym (idl _)) ∙ q ∙ sym (p-kp .p₁∘universal)
```

# Categories with all pullbacks

We also provide a helper module for working with categories that have all
pullbacks.

```agda
has-pullbacks : ∀ {o ℓ} → Precategory o ℓ → Type _
has-pullbacks C = ∀ {A B X} (f : Hom A X) (g : Hom B X) → Pullback C f g
  where open Precategory C

module Pullbacks
  {o ℓ}
  (C : Precategory o ℓ)
  (all-pullbacks : has-pullbacks  C)
  where
  open Precategory C
  module pullback {x y z} (f : Hom x z) (g : Hom y z) =
    Pullback (all-pullbacks f g)

  Pb : ∀ {x y z} → Hom x z → Hom y z → Ob
  Pb = pullback.apex
```

## Stability {defines="pullback-stability pullback-stable"}

Pullbacks, in addition to their nature as limits, serve as the way of
"[[changing the base|pullback functor]]" of a family of objects: if we
think of an arrow
$f : A \to B$ as encoding the data of a family over $B$ (think of the
special case where $A = \Sigma_{x : A} F(x)$, and $f = \pi_1$), then we
can think of pulling back $f$ along $g : X \to B$ as "the universal
solution to making $f$ a family over $X$, via $g$". One way of making
this intuition formal is through the [[fundamental fibration]] of a
category with pullbacks.

In that framing, there is a canonical choice for "the" pullback of an
arrow along another: We put the arrow $f$ we want to pullback on the
right side of the diagram, and the pullback is the left arrow. Using
the type `is-pullback`{.Agda} defined above, the arrow which results
from pulling back is adjacent _to the adjustment_: `is-pullback f⁺ g _ f`.
To help keep this straight, we define what it means for a class of
arrows to be **stable under pullback**: If `f` has a given property,
then so does `f⁺`, for any pullback of `f`.

<!--
```agda
module _ {o ℓ} (C : Precategory o ℓ) where
  open Cat.Reasoning C
```
-->

```agda
  is-pullback-stable
    : ∀ {ℓ'} → (∀ {a b} → Hom a b → Type ℓ') → Type _
  is-pullback-stable P =
    ∀ {p A B X} (f : Hom A B) (g : Hom X B) {f⁺ : Hom p X} {p2}
    → P f → is-pullback C f⁺ g p2 f → P f⁺
```
