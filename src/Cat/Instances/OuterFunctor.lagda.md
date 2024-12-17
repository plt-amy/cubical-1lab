<!--
```agda
open import Cat.Diagram.Product.Solver
open import Cat.Internal.Functor.Outer
open import Cat.Diagram.Product
open import Cat.Prelude

import Cat.Internal.Reasoning
import Cat.Internal.Base
import Cat.Reasoning
```
-->

```agda
module Cat.Instances.OuterFunctor
  {o ℓ} {C : Precategory o ℓ}
  where
```

<!--
```agda
open Cat.Reasoning C
open Cat.Internal.Base C
open Functor
open Outer-functor
open _=>o_

```
-->

# The category of outer functors

Like most constructions in category theory, [outer functors], and outer
natural transformations between them, can also be regarded as a
category. By a rote calculation, we can define the identity and
composite outer natural transformations.

[outer functors]: Cat.Internal.Functor.Outer.html

<!--
```agda
module _ {ℂ : Internal-cat} where
  open Internal-cat ℂ
```
-->

```agda
  idnto : ∀ {F : Outer-functor ℂ} → F =>o F
  idnto {F = F} .ηo px = px
  idnto {F = F} .ηo-fib _ = refl
  idnto {F = F} .is-naturalo px y f =
    ap (F .P₁ px) (Internal-hom-path refl)
  idnto {F = F} .ηo-nat _ _ = refl

  _∘nto_ : ∀ {F G H : Outer-functor ℂ} → G =>o H → F =>o G → F =>o H
  _∘nto_ α β .ηo x = α .ηo (β .ηo x)
  _∘nto_ α β .ηo-fib px = α .ηo-fib (β .ηo px) ∙ β .ηo-fib px
  _∘nto_ {H = H} α β .is-naturalo px y f =
    ap (α .ηo) (β .is-naturalo px y f)
    ∙ α .is-naturalo (β .ηo px) y (adjusti (sym (β .ηo-fib px)) refl f)
    ∙ ap (H .P₁ _) (Internal-hom-path refl)
  (α ∘nto β) .ηo-nat px σ =
    α .ηo-nat (β .ηo px) σ ∙ ap (α .ηo) (β .ηo-nat px σ)
```

These are almost definitionally a precategory, requiring only an appeal
to extensionality to establish the laws.

<!--
```agda
module _ (ℂ : Internal-cat) where
  open Internal-cat ℂ
```
-->

```agda
  Outer-functors : Precategory (o ⊔ ℓ) (o ⊔ ℓ)
  Outer-functors .Precategory.Ob = Outer-functor ℂ
  Outer-functors .Precategory.Hom = _=>o_
  Outer-functors .Precategory.Hom-set _ _ = hlevel 2
  Outer-functors .Precategory.id = idnto
  Outer-functors .Precategory._∘_ = _∘nto_
  Outer-functors .Precategory.idr α = Outer-nat-path (λ _ → refl)
  Outer-functors .Precategory.idl α = Outer-nat-path (λ _ → refl)
  Outer-functors .Precategory.assoc α β γ = Outer-nat-path (λ _ → refl)
```

<!--
```agda
module _ (prods : has-products C) (ℂ : Internal-cat) where
  open Internal-cat ℂ
  open Binary-products C prods
```
-->

## The constant-functor functor

If $\cC$ is a category with products, and $\bC$ is a category internal
to $\cC$, then we can construct a _constant outer-functor functor_ $\cC
\to \rm{Out}(\bC)$.

```agda
  Δo : Functor C (Outer-functors ℂ)
  Δo .F₀ = ConstO prods
  Δo .F₁ = const-nato prods
  Δo .F-id = Outer-nat-path λ px →
    ap₂ ⟨_,_⟩ (idl _) refl
    ∙ sym (⟨⟩∘ px)
    ∙ eliml ⟨⟩-η
  Δo .F-∘ f g = Outer-nat-path λ px →
    ⟨ (f ∘ g) ∘ π₁ ∘ px , π₂ ∘ px ⟩                                        ≡⟨ products! prods ⟩
    ⟨ f ∘ π₁ ∘ ⟨ g ∘ π₁ ∘ px , π₂ ∘ px ⟩ , π₂ ∘ ⟨ g ∘ π₁ ∘ px , π₂ ∘ px ⟩ ⟩ ∎
```
