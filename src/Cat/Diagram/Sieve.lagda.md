<!--
```agda
open import Cat.Instances.Functor
open import Cat.Instances.Slice
open import Cat.Functor.Hom
open import Cat.Prelude

open import Data.Power

import Cat.Reasoning

open /-Obj
```
-->

```agda
module Cat.Diagram.Sieve where
```

<!--
```agda
module _ {o κ : _} (C : Precategory o κ) (c : ⌞ C ⌟) where
  private module C = Precategory C
```
-->

# Sieves {defines="sieve"}

Given a category $\cC$, a **sieve** on an object $c$ Is a subset of
the maps $a \to c$ closed under composition: If $f \in S$, then $(f
\circ g) \in S$. The data of a sieve on $c$ corresponds to the data of a
subobject of $\yo(c)$, considered as an object of $\psh(\cC)$.

Here, the subset is represented as the function `arrows`{.agda}, which,
given an arrow $f : a \to c$ (and its domain), yields a proposition
representing inclusion in the subset.

```agda
  record Sieve : Type (o ⊔ κ) where
    no-eta-equality
    field
      arrows : ∀ {y} → ℙ (C.Hom y c)
      closed : ∀ {y z f} (hf : f ∈ arrows) (g : C.Hom y z) → (f C.∘ g) ∈ arrows
  open Sieve public
```

The `maximal`{.agda} sieve on an object is the collection of _all_ maps
$a \to c$; It represents the identity map $\yo(c) \to \yo(c)$ as a
subfunctor. A [$\kappa$-small] family of sieves can be intersected (the
underlying predicate is the "$\kappa$-ary conjunction" $\Pi$ --- the
universal quantifier), and this represents a wide pullback of
subobjects.

[$\kappa$-small]: 1Lab.intro.html#universes-and-size-issues

<!--
```agda
module _ {o ℓ : _} {C : Precategory o ℓ} where
  private
    module C   = Cat.Reasoning C
    module PSh = Cat.Reasoning (PSh ℓ C)
    open Precategory C

  Sieve-path : ∀ {c} {x y : Sieve C c} → Path (∀ {y} → ℙ (C.Hom y c)) (x .arrows) (y .arrows) → x ≡ y
  Sieve-path {x = x} {y} p i .arrows = p i
  Sieve-path {x = x} {y} p i .closed {f = f} hf g =
    is-prop→pathp (λ i → fun-is-hlevel {A = ⌞ p i f ⌟} 1 (p i (f ∘ g) .is-tr)) (λ w → x .closed w g) (λ w → y .closed w g) i hf

  instance
    hom∈Sieve : ∀ {c d} → Membership (C.Hom d c) (Sieve C c) _
    hom∈Sieve = record { _∈_ = λ x S → x ∈ S .Sieve.arrows }

    slice∈Sieve : ∀ {c} → Membership (/-Obj {C = C} c) (Sieve C c) _
    slice∈Sieve = record { _∈_ = λ x S → x .map ∈ S }

    Inclusion-sieve : ∀ {U} → Inclusion (Sieve C U) _
    Inclusion-sieve {U} = record { _⊆_ = λ S T → ∀ {V} (h : Hom V U) → h ∈ S → h ∈ T }

    Extensional-sieve : ∀ {ℓr c} ⦃ _ : Extensional (∀ {y} → C.Hom y c → Ω) ℓr ⦄ → Extensional (Sieve C c) ℓr
    Extensional-sieve ⦃ e ⦄ = injection→extensional! Sieve-path e

    H-Level-Sieve : ∀ {c n} → H-Level (Sieve C c) (2 + n)
    H-Level-Sieve = basic-instance 2 $
      embedding→is-hlevel 1 (injective→is-embedding! Sieve-path) (hlevel 2)

  open PSh._↪_
  open _=>_
  open Functor
```
-->

```agda
  maximal' : ∀ {c} → Sieve C c
  maximal' .arrows x = ⊤Ω
  maximal' .closed g x = tt

  intersect : ∀ {c} {I : Type ℓ} (F : I → Sieve C c) → Sieve C c
  intersect {I = I} F .arrows h = elΩ ((x : I) → h ∈ F x)
  intersect {I = I} F .closed x g = inc λ i → F i .closed (□-out! x i) g
```

<!--
```agda
  _∩S_ : ∀ {U} → Sieve C U → Sieve C U → Sieve C U
  (S ∩S T) .arrows f = S .arrows f ∧Ω T .arrows f
  (S ∩S T) .closed (Sf , Tf) g = S .closed Sf g , T .closed Tf g
```
-->

## Representing subfunctors {defines="sieves-as-presheaves"}

Let $S$ be a sieve on $\cC$. We show that it determines a presheaf
$S'$, and that this presheaf admits a monic natural transformation $S'
\mono \yo(c)$. The functor determined by a sieve $S$ sends each object
$d$ to the set of arrows $d \xto{f} c$ s.t. $f \in S$; The functorial
action is given by composition, as with the $\hom$ functor.

```agda
  to-presheaf : ∀ {c} → Sieve C c → PSh.Ob
  to-presheaf {c} sieve .F₀ d = el! (Σ[ f ∈ C.Hom d c ] (f ∈ sieve))
  to-presheaf sieve .F₁ f (g , s) = g C.∘ f , sieve .closed s _
```

<!--
```agda
  to-presheaf sieve .F-id    = funext λ _ → Σ-prop-path! (C.idr _)
  to-presheaf sieve .F-∘ f g = funext λ _ → Σ-prop-path! (C.assoc _ _ _)
```
-->

That this functor is a subobject of $\yo$ follows straightforwardly: The
injection map $S' \mono \yo(c)$ is given by projecting out the first
component, which is an embedding (since "being in a sieve" is a
proposition). Since natural transformations are monic if they are
componentwise monic, and embeddings are monic, the result follows.

```agda
  to-presheaf↪よ : ∀ {c} {S : Sieve C c} → to-presheaf S PSh.↪ よ₀ C c
  to-presheaf↪よ {S} .mor .η x (f , _) = f
  to-presheaf↪よ {S} .mor .is-natural x y f = refl
  to-presheaf↪よ {S} .monic g h path = ext λ i x → Σ-prop-path! (unext path i x)
```

## Pullback of sieves {defines="pullback-sieve"}

If we have a sieve $R$ on $U$, and any morphism $f : V \to U$, then
there is a natural way to restrict the $h_i$ to a sieve on $V$: a
morphism $g : V_i \to V$ belongs to the restriction if the composite $fg
: V_i \to V \to U$ belongs to $R$. We refer to the resulting sieve as
the **pullback of $R$ along $f$**, and write it $f^*(R)$.

```agda
  pullback : ∀ {u v} → C.Hom v u → Sieve C u → Sieve C v
  pullback f s .arrows h = el (f C.∘ h ∈ s) (hlevel 1)
  pullback f s .closed hf g = subst (_∈ s) (sym (C.assoc f _ g)) (s .closed hf g)
```

If we consider the collection of "sieves on $U$" as varying along $U$ as
a parameter, the pullback operation becomes functorial. Since it is
contravariant, this means that the assignment $U \mapsto
\operatorname{Sieves}(U)$ is *itself* a presheaf on $U$.

```agda
  abstract
    pullback-id : ∀ {c} {s : Sieve C c} → pullback C.id s ≡ s
    pullback-id {s = s} = ext λ h → biimp
      (subst (_∈ s) (C.idl h))
      (subst (_∈ s) (sym (C.idl h)))

    pullback-∘
      : ∀ {u v w} {f : C.Hom w v} {g : C.Hom v u} {R : Sieve C u}
      → pullback (g C.∘ f) R ≡ pullback f (pullback g R)
    pullback-∘ {f = f} {g} {R = R} = ext λ h → biimp
      (subst (_∈ R) (sym (C.assoc g f h)))
      (subst (_∈ R) (C.assoc g f h))
```

This presheaf has an important universal property: the natural
transformations $X \to \operatorname{Sieves}$ correspond naturally to
the [[subobjects|poset of subobjects]] of $X$. Categorically, we say
that $\operatorname{Sieves}$ is a **subobject classifier** in the
category $\psh(\cC)$.

```agda
  Sieves : Functor (C ^op) (Sets (o ⊔ ℓ))
  Sieves .F₀ U .∣_∣ = Sieve C U
  Sieves .F₀ U .is-tr = hlevel 2
  Sieves .F₁ = pullback
  Sieves .F-id    = funext λ x → pullback-id
  Sieves .F-∘ f g = funext λ x → pullback-∘
```

## Generated sieves

Often, it's more practical to define a *family* of maps, and to obtain a
sieve from this family after the fact. To this end, we define a type
`Cover`{.Agda} for families of maps into a common codomain, paired with
their indexing type.

<!--
```agda
module _ {o ℓ} (C : Precategory o ℓ) where
  open Precategory C using (Hom)
```
-->

```agda
  record Cover (U : ⌞ C ⌟) ℓ' : Type (o ⊔ ℓ ⊔ lsuc ℓ') where
    field
      {index}  : Type ℓ'
      {domain} : index → ⌞ C ⌟
      map      : ∀ i → Hom (domain i) U
```

<!--
```agda
open Cover

module _ {o ℓ} {C : Precategory o ℓ} where
  private module C = Cat.Reasoning C
  instance
    Underlying-Cover : ∀ {ℓ' U} → Underlying (Cover C U ℓ')
    Underlying-Cover = record { ⌞_⌟ = index }
```
-->

The **sieve generated by a cover** $(f_i : V_i \to U)_{i : I}$ is the
collection of maps that factor through at least one of the $f_i$, i.e.,
for a map $g : W \to U$, it is the proposition
$$
\exists (i : I) (h : W \to V_i).~ f_i \circ h = g
$$.

```agda
  cover→sieve : ∀ {ℓ' U} → Cover C U ℓ' → Sieve C U
  cover→sieve {U = U} f .arrows {W} g = elΩ do
    Σ[ i ∈ f ] Σ[ h ∈ C.Hom W (f .domain i) ] (f .map i C.∘ h ≡ g)
  cover→sieve f .closed = rec! λ i h p g → inc (i , h C.∘ g , C.pulll p)
```

Since the primary purpose of `Cover`{.Agda} is to present sieves, we
register an instance of the `⟦⟧-notation`{.Agda} class, so that we can
write `⟦ cov ⟧` instead of `cover→sieve cov`.

<!--
```agda
  map→sieve : ∀ {V U} → C.Hom V U → Sieve C U
  map→sieve f .arrows g = elΩ (Σ[ h ∈ C.Hom _ _ ] (f C.∘ h ≡ g))
  map→sieve f .closed = rec! λ g p h → inc (g C.∘ h , C.pulll p)

  instance
    ⟦⟧-Cover : ∀ {ℓ' U} → ⟦⟧-notation (Cover C U ℓ')
    ⟦⟧-Cover = brackets _ cover→sieve
```
-->
