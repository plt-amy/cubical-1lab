---
description: |
  Discrete cocartesian fibrations.
---
<!--
```agda
open import Cat.Displayed.Cocartesian
open import Cat.Displayed.Functor
open import Cat.Instances.Functor
open import Cat.Displayed.Fibre
open import Cat.Displayed.Base
open import Cat.Displayed.Path
open import Cat.Prelude

import Cat.Displayed.Reasoning
import Cat.Displayed.Morphism
import Cat.Reasoning
```
-->
```agda
module Cat.Displayed.Cocartesian.Discrete where
```

<!--
```agda
open Cocartesian-fibration
open Cocartesian-lift
open is-cocartesian
```
-->

# Discrete cocartesian fibrations

:::{.definition #discrete-cocartesian-fibration alias="discrete-opfibration"}
A **discrete cocartesian fibration** or **discrete opfibration** is a
[[displayed category]] that satisfies the dual lifting property of a
[[discrete cartesian fibration]]. Explicitly, a displayed category
$\cE \liesover \cB$ is a discrete cocartesian fibration if:

- Every type of displayed objects is a set.
- For each left corner

~~~{.quiver}
\[\begin{tikzcd}
  {x'} & \\
  x & {y\text{,}}
  \arrow[lies over, from=1-1, to=2-1]
  \arrow["f"', from=2-1, to=2-2]
\end{tikzcd}\]
~~~

there is a contractible space of objects $y'$ equipped with
maps $x' \to_{f} y'$.

:::


<!--
```agda
module _ {o ℓ o' ℓ'} {B : Precategory o ℓ} (E : Displayed B o' ℓ') where
  private
    module B = Cat.Reasoning B
    module E = Displayed E
    open Cat.Displayed.Reasoning E
    open Cat.Displayed.Morphism E
    open Displayed E
```
-->

```agda
  record is-discrete-cocartesian-fibration : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
    field
      fibre-set : ∀ x → is-set E.Ob[ x ]
      cocart-lift
        : ∀ {x y} (f : B.Hom x y) (x' : E.Ob[ x ])
        → is-contr (Σ[ y' ∈ E.Ob[ y ] ] E.Hom[ f ] x' y')
```


We will denote the canonical lift of $f$ to $y'$ as
$$
\iota_{f, x'}^{*} : x' \to_{f} f_{!}(x')
$$

```agda
    _^!_ : ∀ {x y} → (f : B.Hom x y) → (x' : E.Ob[ x ]) → E.Ob[ y ]
    f ^! x' = cocart-lift f x' .centre .fst

    ι! : ∀ {x y} → (f : B.Hom x y) → (x' : E.Ob[ x ]) → E.Hom[ f ] x' (f ^! x')
    ι! f x' = cocart-lift f x' .centre .snd
```

## Basic properties of discrete cocartesian fibrations

Discrete cocartesian fibrations are formally dual to discrete cartesian
fibrations, so they enjoy many of the same basic properties.
Many of the proofs are functionally identical, so we will only provide
brief commentary, and direct interested readers to the
[corresponding section] in the page for discrete cartesian fibrations
for details.

[corresponding section]: Cat.Displayed.Cartesian.Discrete.html#basic-properties-of-discrete-cartesian-fibrations

First, note that the type of morphisms in a discrete cocartesian fibration
is always a [[proposition]].

```agda
    Hom[]-is-prop : ∀ {x y x' y'} {f : B.Hom x y} → is-prop (Hom[ f ] x' y')
    Hom[]-is-prop {x' = x'} {y' = y'} {f = f} f' f'' =
      Σ-inj-set (fibre-set _) $
      is-contr→is-prop (cocart-lift f x') (y' , f') (y' , f'')
```

Additionally, morphisms $x' \to_{f} y'$ in a discrete cocartesian fibration
are equivalent to proofs that $f_{!}(x') = y'$.

```agda
    opaque
      ^!-lift
        : ∀ {x y x' y'}
        → (f : B.Hom x y)
        → Hom[ f ] x' y'
        → f ^! x' ≡ y'

      ^!-hom
        : ∀ {x y x' y'}
        → (f : B.Hom x y)
        → f ^! x' ≡ y'
        → Hom[ f ] x' y'

      ^!-hom-is-equiv
        : ∀ {x y x' y'}
        → (f : B.Hom x y)
        → is-equiv (^!-hom {x' = x'} {y' = y'} f)
```

<details>
<summary>The proofs are formally dual to the cartesian case, so we omit
the details.
</summary>

```agda

      ^!-lift {x' = x'} {y' = y'} f f' =
        ap fst $ cocart-lift f x' .paths (y' , f')

      ^!-hom {x' = x'} {y' = y'} f p =
        hom[ B.idl f ] $
          subst (λ x' → Hom[ B.id ] x' y') (sym p) id' ∘' ι! f x'

      ^!-hom-is-equiv f =
        is-iso→is-equiv $
        iso (^!-lift f)
          (λ _ → Hom[]-is-prop _ _)
          (λ _ → fibre-set _ _ _ _ _)
```
</details>

## Functoriality of lifts

The (necessarily unique) choice of lifts in a discrete cocartesian fibration
are functorial. Unlike the cartesian case, discrete cartesian fibrations
are *covariantly* functorial; this asymmetry is an artifact of duality.

```agda
    ^!-id
      : ∀ {x} (x' : Ob[ x ])
      → B.id ^! x' ≡ x'
    ^!-id x' = ^!-lift B.id id'

    ^!-∘
      : ∀ {x y z}
      → (f : B.Hom y z) (g : B.Hom x y) (x' : Ob[ x ])
      → (f B.∘ g) ^! x' ≡ f ^! (g ^! x')
    ^!-∘ f g x' = ^!-lift (f B.∘ g) (ι! f (g ^! x') ∘' ι! g x')
```

## Invertible maps in discrete cocartesian fibrations

Let $f : x \to y$ be an [[invertible]] morphism of $\cB$. If $\cE$
is a discrete cocartesian fibration, then every morphism displayed over
$f$ is also invertible.

```agda
    all-invertible[]
      : ∀ {x y x' y'} {f : B.Hom x y}
      → (f' : Hom[ f ] x' y')
      → (f⁻¹ : B.is-invertible f)
      → is-invertible[ f⁻¹ ] f'
```

<details>
<summary>The proof is essentially identical to the [cartesian case],
so we omit the details.
</summary>

[cartesian case]: Cat.Displayed.Cartesian.Discrete.html#invertible-maps-in-discrete-cartesian-fibrations

```agda
    all-invertible[] {x' = x'} {y' = y'} {f = f} f' f⁻¹ = f'⁻¹ where
      module f⁻¹ = B.is-invertible f⁻¹
      open is-invertible[_]

      f'⁻¹ : is-invertible[ f⁻¹ ] f'
      f'⁻¹ .inv' =
        ^!-hom f⁻¹.inv $
          f⁻¹.inv ^! y'         ≡˘⟨ ap (f⁻¹.inv ^!_) (^!-lift f f') ⟩
          f⁻¹.inv ^! (f ^! x')  ≡˘⟨ ^!-∘ f⁻¹.inv f x' ⟩
          (f⁻¹.inv B.∘ f) ^! x' ≡⟨ ap (_^! x') f⁻¹.invr ⟩
          B.id ^! x'            ≡⟨ ^!-id x' ⟩
          x'                    ∎
      f'⁻¹ .inverses' .Inverses[_].invl' =
        is-prop→pathp (λ _ → Hom[]-is-prop) _ _
      f'⁻¹ .inverses' .Inverses[_].invr' =
        is-prop→pathp (λ _ → Hom[]-is-prop) _ _
```
</details>

As an easy corollary, we get that all vertical maps in a discrete
fibration are invertible.

```agda
    all-invertible↓
      : ∀ {x x' y'}
      → (f' : Hom[ B.id {x} ] x' y')
      → is-invertible↓ f'
    all-invertible↓ f' = all-invertible[] f' B.id-invertible
```

## Cocartesian maps in discrete cocartesian fibrations

As the name suggests, every morphism in a discrete cocartesian fibration
is [[cocartesian|cocartesian-morphism]]. Note that as every hom set in a
discrete cocartesian fibration is propositional, so we just
need to establish the existence portion of the universal property.

```agda
    all-cocartesian
      : ∀ {x y x' y'} {f : B.Hom x y}
      → (f' : Hom[ f ] x' y')
      → is-cocartesian E f f'
    all-cocartesian f' .commutes _ _ = Hom[]-is-prop _ _
    all-cocartesian f' .unique _ _ = Hom[]-is-prop _ _
```

<details>
<summary>The argument is almost identical to the proof that [all morphisms
in discrete cartesian fibrations are cartesian], so we suppress the details.
</summary>

[all morphisms in discrete cartesian fibrations are cartesian]: Cat.Displayed.Cartesian.Discrete.html#cartesian-maps-in-discrete-fibrations

```agda
    all-cocartesian {x' = x'} {y' = y'} {f = f} f' .universal {u' = u'} g h' =
      ^!-hom g $
        g ^! y'         ≡˘⟨ ap (g ^!_) (^!-lift f f') ⟩
        g ^! (f ^! x')  ≡˘⟨ ^!-∘ g f x' ⟩
        (g B.∘ f) ^! x' ≡⟨ ^!-lift (g B.∘ f) h' ⟩
        u'              ∎
```
</details>

## Discrete cocartesian fibrations are cocartesian

To prove that discrete cocartesian fibrations deserve the name
_fibration_, we prove that any discrete fibration is a [[cocartesian
fibration]]. Luckily, we already have all the pieces at hand: every morphism
in $\cE$ is cocartesian, so all we need to is to exhibit a lift, and
by our assumption, all such lifts exist!

```agda
  discrete→cocartesian
    : is-discrete-cocartesian-fibration
    → Cocartesian-fibration E
  discrete→cocartesian dfib = cocart-fib where
    open is-discrete-cocartesian-fibration dfib

    cocart-fib : Cocartesian-fibration E
    cocart-fib .has-lift f x' .y' = f ^! x'
    cocart-fib .has-lift f x' .lifting = ι! f x'
    cocart-fib .has-lift f x' .cocartesian = all-cocartesian (ι! f x')
```
