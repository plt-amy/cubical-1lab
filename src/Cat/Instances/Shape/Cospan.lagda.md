<!--
```agda
open import Cat.Prelude
open import Cat.Finite

open import Data.Fin.Product
open import Data.Fin.Finite
open import Data.Dec.Base
open import Data.Fin
```
-->

```agda
module Cat.Instances.Shape.Cospan where
```


# The "cospan" category

A _cospan_ in a category $\cC$ is a pair of morphisms $a \xto{f} c
\xot{g} b$ with a common codomain. A [limit] over a diagram with cospan
shape is called a [[pullback]]. Correspondingly, to encode such
diagrams, we have a "cospan category" $\bull \to \bull \ot \bull$. The
dual of this category, which looks like $\bull \ot \bull \to \bull$, is
the "span" category. Colimits over a span are called [pushouts].

[limit]: Cat.Diagram.Limit.Base.html
[pushouts]: Cat.Diagram.Pushout.html

```agda
data Cospan-ob ℓ : Type ℓ where
  cs-a cs-b cs-c : Cospan-ob ℓ

Cospan-hom : ∀ {ℓ ℓ'} → Cospan-ob ℓ → Cospan-ob ℓ → Type ℓ'
Cospan-hom cs-a cs-a = Lift _ ⊤ -- identity on a
Cospan-hom cs-a cs-b = Lift _ ⊥ -- no maps a → b
Cospan-hom cs-a cs-c = Lift _ ⊤ -- one map a → c
Cospan-hom cs-b cs-a = Lift _ ⊥ -- no maps b → a
Cospan-hom cs-b cs-b = Lift _ ⊤ -- identity on b
Cospan-hom cs-b cs-c = Lift _ ⊤ -- one map b → c
Cospan-hom cs-c cs-a = Lift _ ⊥ -- no maps c → a
Cospan-hom cs-c cs-b = Lift _ ⊥ -- no maps c → b
Cospan-hom cs-c cs-c = Lift _ ⊤ -- identity on c

·→·←· ·←·→· : ∀ {a b} → Precategory a b
```

<!--
```agda
·→·←· = precat where
  open Precategory

  compose : ∀ {a b c} → Cospan-hom b c → Cospan-hom a b → Cospan-hom a c
  compose {cs-a} {cs-a} {cs-a} (lift tt) (lift tt) = lift tt
  compose {cs-a} {cs-a} {cs-c} (lift tt) (lift tt) = lift tt
  compose {cs-a} {cs-c} {cs-c} (lift tt) (lift tt) = lift tt
  compose {cs-b} {cs-b} {cs-b} (lift tt) (lift tt) = lift tt
  compose {cs-b} {cs-b} {cs-c} (lift tt) (lift tt) = lift tt
  compose {cs-b} {cs-c} {cs-c} (lift tt) (lift tt) = lift tt
  compose {cs-c} {cs-c} {cs-c} (lift tt) (lift tt) = lift tt

  precat : Precategory _ _
  precat .Ob = Cospan-ob _
  precat .Hom = Cospan-hom
  precat .Hom-set cs-a cs-a _ _ p q i j = lift tt
  precat .Hom-set cs-a cs-c _ _ p q i j = lift tt
  precat .Hom-set cs-b cs-b _ _ p q i j = lift tt
  precat .Hom-set cs-b cs-c _ _ p q i j = lift tt
  precat .Hom-set cs-c cs-c _ _ p q i j = lift tt
  precat .id {cs-a} = lift tt
  precat .id {cs-b} = lift tt
  precat .id {cs-c} = lift tt
  precat ._∘_ = compose
  precat .idr {cs-a} {cs-a} _ i = lift tt
  precat .idr {cs-a} {cs-c} _ i = lift tt
  precat .idr {cs-b} {cs-b} _ i = lift tt
  precat .idr {cs-b} {cs-c} _ i = lift tt
  precat .idr {cs-c} {cs-c} _ i = lift tt
  precat .idl {cs-a} {cs-a} _ i = lift tt
  precat .idl {cs-a} {cs-c} _ i = lift tt
  precat .idl {cs-b} {cs-b} _ i = lift tt
  precat .idl {cs-b} {cs-c} _ i = lift tt
  precat .idl {cs-c} {cs-c} _ i = lift tt
  precat .assoc {cs-a} {cs-a} {cs-a} {cs-a} _ _ _ i = lift tt
  precat .assoc {cs-a} {cs-a} {cs-a} {cs-c} _ _ _ i = lift tt
  precat .assoc {cs-a} {cs-a} {cs-c} {cs-c} _ _ _ i = lift tt
  precat .assoc {cs-a} {cs-c} {cs-c} {cs-c} _ _ _ i = lift tt
  precat .assoc {cs-b} {cs-b} {cs-b} {cs-b} _ _ _ i = lift tt
  precat .assoc {cs-b} {cs-b} {cs-b} {cs-c} _ _ _ i = lift tt
  precat .assoc {cs-b} {cs-b} {cs-c} {cs-c} _ _ _ i = lift tt
  precat .assoc {cs-b} {cs-c} {cs-c} {cs-c} _ _ _ i = lift tt
  precat .assoc {cs-c} {cs-c} {cs-c} {cs-c} _ _ _ i = lift tt

·←·→· = ·→·←· ^op

instance
  Finite-Cospan-ob : ∀ {ℓ} → Finite (Cospan-ob ℓ)
  Finite-Cospan-ob = inc (Equiv→listing (Equiv.inverse (Iso→Equiv i)) (Listing-Fin {n = 3})) where
    i : Iso _ _
    i .fst cs-a = 0
    i .fst cs-b = 1
    i .fst cs-c = 2
    i .snd .is-iso.inv = indexₚ (cs-a , cs-b , cs-c , tt)
    i .snd .is-iso.rinv = indexₚ (refl , refl , refl , tt)
    i .snd .is-iso.linv cs-a = refl
    i .snd .is-iso.linv cs-b = refl
    i .snd .is-iso.linv cs-c = refl

·→·←·-finite : ∀ {a b} → is-finite-precategory (·→·←· {a} {b})
·→·←·-finite = finite-cat-hom λ where
  cs-a cs-a → auto
  cs-a cs-b → auto
  cs-a cs-c → auto
  cs-b cs-a → auto
  cs-b cs-b → auto
  cs-b cs-c → auto
  cs-c cs-a → auto
  cs-c cs-b → auto
  cs-c cs-c → auto
```
-->

Converting a pair of morphisms with common codomain to a cospan-shaped
diagram is straightforward:

```agda
module _ x y {o ℓ} {C : Precategory o ℓ} where
  open Precategory C
  open Functor

  cospan→cospan-diagram : ∀ {a b c} → Hom a c → Hom b c → Functor (·→·←· {x} {y}) C
  cospan→cospan-diagram f g = funct where
    funct : Functor _ _
    funct .F₀ cs-a = _
    funct .F₀ cs-b = _
    funct .F₀ cs-c = _
    funct .F₁ {cs-a} {cs-c} _ = f
    funct .F₁ {cs-b} {cs-c} _ = g
```

<!--
```agda
    funct .F₁ {cs-a} {cs-a} _ = _
    funct .F₁ {cs-b} {cs-b} _ = _
    funct .F₁ {cs-c} {cs-c} _ = _
    funct .F-id {cs-a} = refl
    funct .F-id {cs-b} = refl
    funct .F-id {cs-c} = refl
    funct .F-∘ {cs-a} {cs-a} {cs-a} _ _ i = idl id (~ i)
    funct .F-∘ {cs-a} {cs-a} {cs-c} _ _ i = idr f (~ i)
    funct .F-∘ {cs-a} {cs-c} {cs-c} _ _ i = idl f (~ i)
    funct .F-∘ {cs-b} {cs-b} {cs-b} _ _ i = idl id (~ i)
    funct .F-∘ {cs-b} {cs-b} {cs-c} _ _ i = idr g (~ i)
    funct .F-∘ {cs-b} {cs-c} {cs-c} _ _ i = idl g (~ i)
    funct .F-∘ {cs-c} {cs-c} {cs-c} _ _ i = idl id (~ i)

  span→span-diagram : ∀ {a b c} → Hom a b → Hom a c → Functor (·←·→· {x} {y}) C
  span→span-diagram {a} {b} {c} f g = funct where
    funct : Functor _ _
    funct .F₀ cs-a = _
    funct .F₀ cs-b = _
    funct .F₀ cs-c = _
    funct .F₁ {cs-a} {cs-a} _ = id
    funct .F₁ {cs-b} {cs-b} _ = id
    funct .F₁ {cs-c} {cs-a} _ = g
    funct .F₁ {cs-c} {cs-b} _ = f
    funct .F₁ {cs-c} {cs-c} _ = id
    funct .F-id {cs-a} = refl
    funct .F-id {cs-b} = refl
    funct .F-id {cs-c} = refl
    funct .F-∘ {cs-a} {cs-a} {cs-a} _ _ i = idl id (~ i)
    funct .F-∘ {cs-b} {cs-b} {cs-b} _ _ i = idl id (~ i)
    funct .F-∘ {cs-c} {cs-a} {cs-a} _ _ i = idl g (~ i)
    funct .F-∘ {cs-c} {cs-b} {cs-b} _ _ i = idl f (~ i)
    funct .F-∘ {cs-c} {cs-c} {cs-a} _ _ i = idr g (~ i)
    funct .F-∘ {cs-c} {cs-c} {cs-b} _ _ i = idr f (~ i)
    funct .F-∘ {cs-c} {cs-c} {cs-c} _ _ i = idr id (~ i)
```
-->
