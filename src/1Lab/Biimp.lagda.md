---
description: |
  Biimplications.
---
<!--
```agda
open import 1Lab.Reflection.Record
open import 1Lab.Extensionality
open import 1Lab.HLevel.Closure
open import 1Lab.HLevel
open import 1Lab.Equiv
open import 1Lab.Type

open import Meta.Invariant
```
-->
```agda
module 1Lab.Biimp where
```

# Biimplications

:::{.definition #biimplication}
A **biimplication** $A \leftrightarrow B$ between a pair of types $A, B$
is a pair of functions $A \to B, B \to A$.
:::

```agda
record _↔_ {ℓ} {ℓ'} (A : Type ℓ) (B : Type ℓ') : Type (ℓ ⊔ ℓ') where
  no-eta-equality
  constructor biimp
  field
    to : A → B
    from : B → A

open _↔_
```

<!--
```agda
private variable
  ℓ ℓ' : Level
  A B C : Type ℓ
```
-->

<!--
```agda
module Biimp (f : A ↔ B) where
  open _↔_ f public
```
-->

If $A$ and $B$ [[n-types]], then the type of biimplications $A \leftrightarrow B$
is also an n-type.

<!--
```agda
private unquoteDecl eqv = declare-record-iso eqv (quote _↔_)
```
-->

```agda
↔-is-hlevel : ∀ n → is-hlevel A n → is-hlevel B n → is-hlevel (A ↔ B) n
↔-is-hlevel n A-hl B-hl =
  Iso→is-hlevel n eqv $
  ×-is-hlevel n
    (Π-is-hlevel n λ _ → B-hl)
    (Π-is-hlevel n λ _ → A-hl)
```

<!--
```agda
instance
  H-Level-↔
    : ∀ {n}
    → ⦃ _ : H-Level A n ⦄ ⦃ _ : H-Level B n ⦄
    → H-Level (A ↔ B) n
  H-Level-↔ {n = n} .H-Level.has-hlevel =
    ↔-is-hlevel n (hlevel n) (hlevel n)

instance
  Extensional-↔
    : ∀ {ℓr}
    → ⦃ _ : Extensional ((A → B) × (B → A)) ℓr ⦄
    → Extensional (A ↔ B) ℓr
  Extensional-↔ ⦃ e ⦄ = iso→extensional eqv e
```
-->

## Working with biimplications

There is an identity biimplication, and biimplications compose.

```agda
id↔ : A ↔ A
id↔ .to = id
id↔ .from = id

_∙↔_ : A ↔ B → B ↔ C → A ↔ C
(f ∙↔ g) .to = g .to ∘ f .to
(f ∙↔ g) .from = f .from ∘ g .from
```

Moreover, every biimplication $A \leftrightarrow B$ induces a biimplication
$B \leftrightarrow A$.

```agda
_↔⁻¹ : A ↔ B → B ↔ A
(f ↔⁻¹) .to = f .from
(f ↔⁻¹) .from = f .to
```

## Biimplications and equivalences

Every [[equivalence]] is a biimplication.

```agda
equiv→biimp : A ≃ B → A ↔ B
equiv→biimp f .to = Equiv.to f
equiv→biimp f .from = Equiv.from f
```

:::{.definition #logical-equivalence}

Every biimplication between [[propositions]] is an [[equivalence]].
In light of this, biimplications are often referred to as
**logical equivalences**.

```agda
biimp→equiv : is-prop A → is-prop B → A ↔ B → A ≃ B
biimp→equiv A-prop B-prop f =
  prop-ext A-prop B-prop (f .to) (f .from)
```
:::

<!--
```agda
infix 21 _↔_
infixr 30 _∙↔_
infix 31 _↔⁻¹
```
-->
