<!--
```agda
open import 1Lab.Prelude

open import Algebra.Group.Homotopy.BAut

open import Data.Maybe.Properties
open import Data.Set.Coequaliser
open import Data.Fin.Properties
open import Data.Fin.Closure
open import Data.Fin.Base
open import Data.Nat.Base
open import Data.Maybe
open import Data.Dec
open import Data.Irr
open import Data.Sum
```
-->

```agda
module Data.Fin.Finite where
```

# Finite types

This module pieces together a couple of pre-existing constructions: In
terms of the [[standard finite sets|standard finite set]] (which are
defined for natural numbers $n$) and [deloopings of automorphism
groups], we construct the type of finite types. [By univalence], the
space of finite types classifies maps with finite fibres.

[deloopings of automorphism groups]: Algebra.Group.Homotopy.BAut.html
[By univalence]: 1Lab.Univalence.html#object-classifiers

But what does it mean for a type to be finite? A naïve first approach is
to define "$X$ is finite" to mean "$X$ is equipped with $n : \NN$ and $f
: [n] \simeq X$" but this turns out to be _too strong_: This doesn't
just equip the type $X$ with a cardinality, but also with a choice of
total order. Additionally, defined like this, the type of finite types
_is a set_!

```agda
naïve-fin-is-set : is-set (Σ[ X ∈ Type ] Σ[ n ∈ Nat ] Fin n ≃ X)
naïve-fin-is-set = Equiv→is-hlevel 2 Σ-swap₂ $
  Σ-is-hlevel 2 (hlevel 2) λ x → is-prop→is-hlevel-suc {n = 1} $
    is-contr→is-prop $ Equiv-is-contr (Fin x)
```

That's because, as the proof above shows, it's equivalent to the type of
natural numbers: The type

$$
\sum_{X : \ty} \sum_{n : \NN}\ [n] \simeq X
$$

is equivalent to the type

$$
\sum_{n : \NN} \sum_{X : \ty} [n] \simeq X
$$,

and univalence says (rather directly) that the sum of $[n] \simeq X$ as
$X$ ranges over a universe is contractible, so we're left with the type
of natural numbers.

This simply won't do: we want the type of finite sets to be equivalent
to the ([[core]] of the) _category_ of finite sets, where the automorphism
group of $n$ has $n!$ elements, not exactly one element. What we do is
appeal to a basic intuition: A groupoid is the sum over its connected
components, and we have representatives for every connected component
(given by the standard finite sets):

```agda
Fin-type : Type (lsuc lzero)
Fin-type = Σ[ n ∈ Nat ] BAut (Fin n)

Fin-type-is-groupoid : is-hlevel Fin-type 3
Fin-type-is-groupoid = Σ-is-hlevel 3 (hlevel 3) λ _ →
  BAut-is-hlevel (Fin _) 2 (hlevel 2)
```

:::{.definition #finite alias="finite-type finite-set"}
Informed by this, we now express the correct definition of "being
finite", namely, being [[merely]] equivalent to some standard finite
set.  Rather than using Σ types for this, we can set up typeclass
machinery for automatically deriving boring instances of finiteness,
i.e. those that follow directly from the closure properties.
:::

```agda
record Finite {ℓ} (T : Type ℓ) : Type ℓ where
  constructor fin
  field
    {cardinality} : Nat
    enumeration   : ∥ T ≃ Fin cardinality ∥
```

<!--
```agda
  Finite→is-set : is-set T
  Finite→is-set =
    ∥-∥-rec (is-hlevel-is-prop 2) (λ e → Equiv→is-hlevel 2 e (hlevel 2)) enumeration

  instance
    Finite→H-Level : H-Level T 2
    Finite→H-Level = basic-instance 2 Finite→is-set

open Finite ⦃ ... ⦄ using (cardinality; enumeration) public
open Finite using (Finite→is-set) public

instance opaque
  H-Level-Finite : ∀ {ℓ} {A : Type ℓ} {n : Nat} → H-Level (Finite A) (suc n)
  H-Level-Finite = prop-instance {T = Finite _} λ where
    x y i .Finite.cardinality → ∥-∥-out!
      ⦇ Fin-injective (⦇ ⦇ x .enumeration e⁻¹ ⦈ ∙e y .enumeration ⦈) ⦈
      i
    x y i .Finite.enumeration → is-prop→pathp
      {B = λ i → ∥ _ ≃ Fin (∥-∥-out! ⦇ Fin-injective (⦇ ⦇ x .enumeration e⁻¹ ⦈ ∙e y .enumeration ⦈) ⦈ i) ∥}
      (λ _ → squash)
      (x .enumeration) (y .enumeration) i

Finite→Discrete : ∀ {ℓ} {A : Type ℓ} → ⦃ Finite A ⦄ → Discrete A
Finite→Discrete {A = A} ⦃ f ⦄ {x} {y} = rec! go (f .enumeration) where
  open Finite f using (Finite→H-Level)
  go : A ≃ Fin (f .cardinality) → Dec (x ≡ y)
  go e with Equiv.to e x ≡? Equiv.to e y
  ... | yes p = yes (Equiv.injective e p)
  ... | no ¬p = no λ p → ¬p (ap (e .fst) p)

Dec→Finite : ∀ {ℓ} {A : Type ℓ} → is-prop A → Dec A → Finite A
Dec→Finite ap d = fin (inc (Dec→Fin ap d .snd e⁻¹))

Discrete→Finite≡ : ∀ {ℓ} {A : Type ℓ} → Discrete A → {x y : A} → Finite (x ≡ y)
Discrete→Finite≡ d = Dec→Finite (Discrete→is-set d _ _) d

Finite-choice
  : ∀ {ℓ ℓ'} {A : Type ℓ} {B : A → Type ℓ'}
  → ⦃ Finite A ⦄
  → (∀ x → ∥ B x ∥) → ∥ (∀ x → B x) ∥
Finite-choice {B = B} ⦃ fin {sz} e ⦄ k = do
  e ← e
  choose ← finite-choice sz λ x → k (equiv→inverse (e .snd) x)
  pure $ λ x → subst B (equiv→unit (e .snd) x) (choose (e .fst x))

Finite-≃ : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} → ⦃ Finite A ⦄ → A ≃ B → Finite B
Finite-≃ ⦃ fin {n} e ⦄ e' = fin (∥-∥-map (e' e⁻¹ ∙e_) e)

equiv→same-cardinality
  : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} ⦃ fa : Finite A ⦄ ⦃ fb : Finite B ⦄
  → ∥ A ≃ B ∥ → fa .Finite.cardinality ≡ fb .Finite.cardinality
equiv→same-cardinality ⦃ fa ⦄ ⦃ fb ⦄ e = ∥-∥-out! do
  e ← e
  ea ← fa .Finite.enumeration
  eb ← fb .Finite.enumeration
  pure (Fin-injective (ea e⁻¹ ∙e e ∙e eb))

same-cardinality→equiv
  : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} ⦃ fa : Finite A ⦄ ⦃ fb : Finite B ⦄
  → fa .Finite.cardinality ≡ fb .Finite.cardinality → ∥ A ≃ B ∥
same-cardinality→equiv ⦃ fa ⦄ ⦃ fb ⦄ p = do
  ea ← fa .Finite.enumeration
  eb ← fb .Finite.enumeration
  pure (ea ∙e path→equiv (ap Fin p) ∙e eb e⁻¹)

module _ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} ⦃ fb : Finite B ⦄
  (e : ∥ A ≃ B ∥) (f : A → B) where

  Finite-injection→equiv : injective f → is-equiv f
  Finite-injection→equiv inj = ∥-∥-out! do
    e ← e
    eb ← fb .Finite.enumeration
    pure
      $ equiv-cancell (eb .snd)
      $ equiv-cancelr ((eb e⁻¹ ∙e e e⁻¹) .snd)
      $ Fin-injection→equiv _
      $ Equiv.injective (eb e⁻¹ ∙e e e⁻¹) ∘ inj ∘ Equiv.injective eb

  Finite-surjection→equiv : is-surjective f → is-equiv f
  Finite-surjection→equiv surj = ∥-∥-out! do
    e ← e
    eb ← fb .Finite.enumeration
    pure
      $ equiv-cancell (eb .snd)
      $ equiv-cancelr ((eb e⁻¹ ∙e e e⁻¹) .snd)
      $ Fin-surjection→equiv _
      $ ∘-is-surjective (is-equiv→is-surjective (eb .snd))
      $ ∘-is-surjective surj
      $ is-equiv→is-surjective ((eb e⁻¹ ∙e e e⁻¹) .snd)

private variable
  ℓ : Level
  A B : Type ℓ
  P Q : A → Type ℓ
```
-->

```agda
instance
  Finite-Fin : ∀ {n} → Finite (Fin n)
  Finite-⊎ : ⦃ Finite A ⦄ → ⦃ Finite B ⦄ → Finite (A ⊎ B)
  Finite-× : ⦃ Finite A ⦄ → ⦃ Finite B ⦄ → Finite (A × B)
  Finite-Maybe : ⦃ fa : Finite A ⦄ → Finite (Maybe A)

  Finite-Σ
    : {P : A → Type ℓ} → ⦃ Finite A ⦄ → ⦃ ∀ {x} → Finite (P x) ⦄ → Finite (Σ A P)
  Finite-Π
    : {P : A → Type ℓ} → ⦃ Finite A ⦄ → ⦃ ∀ {x} → Finite (P x) ⦄ → Finite (∀ x → P x)

  Finite-⊥ : Finite ⊥
  Finite-⊤ : Finite ⊤
  Finite-Bool : Finite Bool

  Finite-PathP
    : ∀ {A : I → Type ℓ} ⦃ s : Finite (A i1) ⦄ {x y}
    → Finite (PathP A x y)

  Finite-Lift : ∀ {ℓ} → ⦃ Finite A ⦄ → Finite (Lift ℓ A)
```

<!--
```agda
  {-# OVERLAPPING Finite-× #-}

Finite-Fin = fin (inc (_ , id-equiv))

Finite-⊎ {A = A} {B = B} = fin $ do
  aeq ← enumeration {T = A}
  beq ← enumeration {T = B}
  pure (⊎-ap aeq beq ∙e Finite-coproduct)

Finite-× {A = A} {B = B} = fin $ do
  aeq ← enumeration {T = A}
  beq ← enumeration {T = B}
  pure (Σ-ap aeq (λ _ → beq) ∙e Finite-multiply)

Finite-Maybe {A = A} = fin do
  an ← enumeration {T = ⊤ ⊎ A}
  pure (Maybe-is-sum ∙e an)

Finite-Π {A = A} {P = P} ⦃ afin ⦄ ⦃ pfin ⦄ = ∥-∥-out! do
  aeq ← afin .Finite.enumeration
  let
    module aeq = Equiv aeq
    bc : Fin (afin .Finite.cardinality) → Nat
    bc x = pfin {aeq.from x} .Finite.cardinality
  pure $ fin do
    t ← Finite-choice λ x → pfin {x} .Finite.enumeration
    pure (Π-cod≃ t ∙e Π-dom≃ aeq.inverse ∙e Finite-product bc)

Finite-Σ {A = A} {P = P} ⦃ afin ⦄ ⦃ pfin ⦄ = ∥-∥-out! do
  aeq ← afin .Finite.enumeration
  let
    module aeq = Equiv aeq
    bc : Fin (afin .Finite.cardinality) → Nat
    bc x = pfin {aeq.from x} .Finite.cardinality
  pure $ fin do
    t ← Finite-choice λ x → pfin {x} .Finite.enumeration
    pure (Σ-ap-snd t ∙e Σ-ap-fst aeq.inverse e⁻¹ ∙e Finite-sum bc)

Finite-⊥ = fin (inc (Finite-zero-is-initial e⁻¹))
Finite-⊤ = fin (inc (is-contr→≃⊤ Finite-one-is-contr e⁻¹))

Bool≃Fin2 : Bool ≃ Fin 2
Bool≃Fin2 = Iso→Equiv enum where
  enum : Iso Bool (Fin 2)
  enum .fst false = 0
  enum .fst true = 1
  enum .snd .is-iso.inv i with fin-view i
  enum .snd .is-iso.inv _ | zero  = false
  enum .snd .is-iso.inv _ | suc _ = true
  enum .snd .is-iso.rinv i with fin-view i
  enum .snd .is-iso.rinv _ | suc fzero = refl
  enum .snd .is-iso.rinv _ | suc (fin (suc n) ⦃ forget p ⦄) = absurd (¬suc≤0 (≤-peel p))
  enum .snd .is-iso.rinv _ | zero  = refl
  enum .snd .is-iso.linv true = refl
  enum .snd .is-iso.linv false = refl

Finite-Bool = fin (inc Bool≃Fin2)

Finite-PathP = subst Finite (sym (PathP≡Path _ _ _)) (Discrete→Finite≡ Finite→Discrete)

Finite-Lift = Finite-≃ (Lift-≃ e⁻¹)
```
-->

```agda
abstract instance
  Finite-Coeq : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} {f g : A → B} ⦃ _ : Finite A ⦄ ⦃ _ : Finite B ⦄ → Finite (Coeq f g)
```

<!--
```agda
  Finite-Coeq {A = A} {B} {f} {g} ⦃ fin ae ⦄ ⦃ fin be ⦄ = ∥-∥-out! do
    ae ← ae
    be ← be
    let
      f' = Equiv.to be ∘ f ∘ Equiv.from ae
      g' = Equiv.to be ∘ g ∘ Equiv.from ae

      fn : Σ[ n ∈ Nat ] Fin n ≃ Coeq f' g'
      fn = Finite-coequaliser f' g'

    pure (fin {cardinality = fn .fst} (inc (Coeq-ap ae be refl refl ∙e Equiv.inverse (fn .snd))))
```
-->

<!--
```agda
card-zero→empty : ∥ A ≃ Fin 0 ∥ → ¬ A
card-zero→empty ∥e∥ a = rec! (λ e → Fin-absurd (Equiv.to e a)) ∥e∥
```
-->
