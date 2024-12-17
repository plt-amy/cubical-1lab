<!--
```agda
open import Cat.Displayed.Cartesian.Discrete
open import Cat.Diagram.Product.Solver
open import Cat.Displayed.Cartesian
open import Cat.Displayed.Adjoint
open import Cat.Displayed.Functor
open import Cat.Diagram.Product
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Displayed.Instances.Simple as Simple
import Cat.Reasoning
```
-->

```agda
module Cat.Displayed.Instances.CT-Structure
  {o ℓ} (B : Precategory o ℓ)
  (has-prods : ∀ X Y → Product B X Y)
  where

open Precategory B
open Binary-products B has-prods
open Simple B has-prods
```

# CT structures

CT-Structures are a refinement of [simple fibrations], which allow us
to model type theories where the contexts aren't necessarily
representable as types. This is done by defining a `CT-Structure`{.Agda}
(short for Context and Type Structure) on our category of contexts,
which allows us to distinguish certain contexts as denoting types
(for instance, singleton contexts, or possibly the empty context if we
have unit types). A CT-structure is quite simple, consisting of only
a predicate on contexts meant to denote which ones are types, along
with the restriction that there is at least one type, to prevent
the entire structure from becoming degenerate.

[simple fibrations]: Cat.Displayed.Instances.Simple.html
[simple fibration]: Cat.Displayed.Instances.Simple.html

```agda
record CT-Structure (s : Level) : Type (o ⊔ lsuc s) where
  field
    is-tp : Ob → Prop s
    ∃-tp  : ∃[ tp ∈ Ob ] (tp ∈ is-tp)

open CT-Structure
```

We can construct a [[displayed category]] over our category of contexts in
much the same manner as the [simple fibration]; the only difference
is that we restrict the displayed object to objects that the
CT-structure distinguishes as types.


```agda
Simple-ct : ∀ {s} → CT-Structure s → Displayed B (o ⊔ s) ℓ
Simple-ct ct .Displayed.Ob[_] Γ = Σ[ X ∈ Ob ] ∣ is-tp ct X ∣
Simple-ct ct .Displayed.Hom[_] {Γ} {Δ} u X Y = Hom (Γ ⊗₀ X .fst) (Y .fst)
Simple-ct ct .Displayed.Hom[_]-set {Γ} {Δ} u X Y = Hom-set (Γ ⊗₀ X .fst) (Y .fst)
Simple-ct ct .Displayed.id' = π₂
Simple-ct ct .Displayed._∘'_ {f = u} {g = v} f g = f ∘ ⟨ v ∘ π₁ , g ⟩
Simple-ct ct .Displayed.idr' {f = u} f =
  f ∘ ⟨ (id ∘ π₁) , π₂ ⟩ ≡⟨ products! has-prods ⟩
  f                      ∎
Simple-ct ct .Displayed.idl' {f = u} f =
  π₂ ∘ ⟨ u ∘ π₁ , f ⟩ ≡⟨ products! has-prods ⟩
  f                   ∎
Simple-ct ct .Displayed.assoc' {f = u} {g = v} {h = w} f g h =
  f ∘ ⟨ (v ∘ w) ∘ π₁ , g ∘ ⟨ w ∘ π₁ , h ⟩ ⟩ ≡⟨ products! has-prods ⟩
  (f ∘ ⟨ v ∘ π₁ , g ⟩) ∘ ⟨ w ∘ π₁ , h ⟩     ∎
```

# Cartesian maps

If a map is cartesian in the simple fibration, then it is cartesian
in a simple CT fibration.

```agda
Simple-cartesian→Simple-ct-cartesian
  : ∀ {s} {Γ Δ x y} {σ : Hom Γ Δ} {f : Hom (Γ ⊗₀ x) y}
  → (ct : CT-Structure s)
  → (x-tp : x ∈ ct .is-tp) → (y-tp : y ∈ ct .is-tp)
  → is-cartesian Simple σ f
  → is-cartesian (Simple-ct ct) {a' = x , x-tp} {b' = y , y-tp} σ f
Simple-cartesian→Simple-ct-cartesian ct x-tp y-tp cart = ct-cart where
  open is-cartesian

  ct-cart : is-cartesian (Simple-ct ct) _ _
  ct-cart .universal = cart .universal
  ct-cart .commutes = cart .commutes
  ct-cart .unique = cart .unique
```


# Fibration structure

Much like the simple fibration, the simple fibration associated to a
CT-structure also deserves its name.

```agda
open Cartesian-fibration
open Cartesian-lift
open is-cartesian

Simple-ct-fibration : ∀ {s} (ct : CT-Structure s) → Cartesian-fibration (Simple-ct ct)
Simple-ct-fibration ct .has-lift u Y .x' = Y
Simple-ct-fibration ct .has-lift u Y .lifting = π₂
Simple-ct-fibration ct .has-lift u Y .cartesian .universal _ h = h
Simple-ct-fibration ct .has-lift u Y .cartesian .commutes g h = π₂∘⟨⟩
Simple-ct-fibration ct .has-lift u Y .cartesian .unique {m = g} {h' = h} h' p =
  h'                   ≡˘⟨ π₂∘⟨⟩ ⟩
  π₂ ∘ ⟨ g ∘ π₁ , h' ⟩ ≡⟨ p ⟩
  h ∎
```

# Embeddings

There is an evident embedding of the simple fibration associated with a
CT-structure into the simple fibration.

```agda
Simple-ct→Simple
  : ∀ {s} → (ct : CT-Structure s)
  → Vertical-functor (Simple-ct ct) Simple
Simple-ct→Simple ct .Vertical-functor.F₀' = fst
Simple-ct→Simple ct .Vertical-functor.F₁' f = f
Simple-ct→Simple ct .Vertical-functor.F-id' = refl
Simple-ct→Simple ct .Vertical-functor.F-∘' = refl
```

Furthermore, if $\cB$ is (merely) inhabited, then we can construct a
CT-Structure that considers every context a type.

```agda
All-types : ∥ Ob ∥ → CT-Structure lzero
All-types X .is-tp _ = el ⊤ (hlevel 1)
All-types X .∃-tp = ∥-∥-map (λ x → x , tt) X

Simple→Simple-ct : ∀ {X} → Vertical-functor Simple (Simple-ct (All-types X))
Simple→Simple-ct .Vertical-functor.F₀' X = X , tt
Simple→Simple-ct .Vertical-functor.F₁' f = f
Simple→Simple-ct .Vertical-functor.F-id' = refl
Simple→Simple-ct .Vertical-functor.F-∘' = refl
```
