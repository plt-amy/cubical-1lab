<!--
```agda
open import Algebra.Monoid.Category
open import Algebra.Semigroup
open import Algebra.Monoid
open import Algebra.Magma

open import Cat.Monoidal.Instances.Cartesian
open import Cat.Displayed.Univalence.Thin
open import Cat.Functor.Hom.Representable
open import Cat.Functor.FullSubcategory
open import Cat.Diagram.Product.Solver
open import Cat.Functor.Equivalence
open import Cat.Functor.Properties
open import Cat.Instances.Functor
open import Cat.Diagram.Terminal
open import Cat.Diagram.Product
open import Cat.Displayed.Total
open import Cat.Instances.Sets
open import Cat.Functor.Base
open import Cat.Functor.Hom
open import Cat.Prelude

import Cat.Reasoning

open is-monoid renaming (idl to mon-idl ; idr to mon-idr)
```
-->

```agda
module Cat.Monoidal.Diagram.Monoid.Representable where
```

<!--
```agda
-- Under the module header to prevent sorting from ruining the import list
open import Cat.Monoidal.Diagram.Monoid
  renaming (Monoid-on to Monoid-ob;
            is-monoid-hom to internal-monoid-hom)

module _
  {o ℓ} {C : Precategory o ℓ}
  (prod : has-products C)
  (term : Terminal C)
  where

  open Cat.Reasoning C
  open Binary-products C prod
  open Terminal term
  open Monoid-ob
  open internal-monoid-hom
  open Monoid-hom
  open Functor
  open _=>_
  open Total-hom
  open Representation
```
-->

# Externalising monoids as presheaves

Let $\cC$ be a Cartesian monoidal category, and consider a [monoid
object] $(M, \eta, \mu)$ in it. Our first observation is that, despite
the definition of monoid object only referring to (the monoidal
structure on $\cC$) and the object $M$, each set of _generalised
elements_ $X \to M$ also carries the structure of a monoid. The unit
element is given by

$$
X \xto{!} 1 \xto{\eta} M
$$,

and the multiplication map is given by

$$
(X \times X) \xto{\langle f, g \rangle} (M \times M) \xto{\mu} M
$$.

[monoid object]: Cat.Monoidal.Diagram.Monoid.html

<!--
```agda
  private
    C-Monoid : Ob → Type ℓ
    C-Monoid m = Monoid-ob (Cartesian-monoidal prod term) m

    C-Monoid-hom : ∀ {m n} → Hom m n → C-Monoid m → C-Monoid n → Type ℓ
    C-Monoid-hom f m-mon n-mon =
      internal-monoid-hom (Cartesian-monoidal prod term) f m-mon n-mon
```
-->

```agda
  Mon→Hom-mon : ∀ {m} (x : Ob) → C-Monoid m → Monoid-on (Hom x m)
  Mon→Hom-mon {m} x mon = hom-mon where
    has-semigroup : is-semigroup λ f g → mon .μ ∘ ⟨ f , g ⟩
    hom-mon : Monoid-on (Hom x m)

    hom-mon .Monoid-on.identity = mon .η ∘ !
    hom-mon .Monoid-on._⋆_ f g  = mon .μ ∘ ⟨ f , g ⟩
```

<details>
<summary>It's not very hard to show that the monoid laws from the
diagram "relativize" to each $\hom$-set.</summary>

```agda
    hom-mon .Monoid-on.has-is-monoid .has-is-semigroup = has-semigroup
    hom-mon .Monoid-on.has-is-monoid .mon-idl {f} =
      mon .μ ∘ ⟨ mon .η ∘ ! , f ⟩         ≡⟨ products! prod ⟩
      mon .μ ∘ (mon .η ⊗₁ id) ∘ ⟨ ! , f ⟩ ≡⟨ pulll (mon .μ-unitl) ⟩
      π₂ ∘ ⟨ ! , f ⟩                      ≡⟨ π₂∘⟨⟩ ⟩
      f                                   ∎
    hom-mon .Monoid-on.has-is-monoid .mon-idr {f} =
      mon .μ ∘ ⟨ f , mon .η ∘ ! ⟩         ≡⟨ products! prod ⟩
      mon .μ ∘ (id ⊗₁ mon .η) ∘ ⟨ f , ! ⟩ ≡⟨ pulll (mon .μ-unitr) ⟩
      π₁ ∘ ⟨ f , ! ⟩                      ≡⟨ π₁∘⟨⟩ ⟩
      f                                   ∎

    has-semigroup .has-is-magma .has-is-set = Hom-set _ _
    has-semigroup .associative {f} {g} {h} =
      mon .μ ∘ ⟨ f , mon .μ ∘ ⟨ g , h ⟩ ⟩                                            ≡⟨ products! prod ⟩
      mon .μ ∘ (id ⊗₁ mon .μ) ∘ ⟨ f , ⟨ g , h ⟩ ⟩                                    ≡⟨ extendl (mon .μ-assoc) ⟩
      mon .μ ∘ ((mon .μ ⊗₁ id) ∘ ⟨ ⟨ π₁ , π₁ ∘ π₂ ⟩ , π₂ ∘ π₂ ⟩) ∘ ⟨ f , ⟨ g , h ⟩ ⟩ ≡⟨ products! prod ⟩
      mon .μ ∘ ⟨ mon .μ ∘ ⟨ f , g ⟩ , h ⟩                                            ∎
```

</details>

Thinking in terms of $M$'s _internal language_, where we think of the
$\hom$-set $X \to M$ as being the set of "$M$-elements in context $X$",
our observation means that $M$ is a monoid in _any_ context. Under this
interpretation, pre-composition with a map $f : X \to Y$ corresponds to
the _substitution_ operation, mapping terms from the context $Y$ to $X$.

Following this line of thinking, the next thing to interrogate is
whether the monoid operations on terms $Y \to M$ is preserved by
substitution: is precomposition with $f$ a monoid homomorphism? The
answer is yes!

```agda
  precompose-hom-mon-hom
    : ∀ {x y m} {mon : C-Monoid m}
    → (f : Hom x y)
    → Monoid-hom (Mon→Hom-mon y mon) (Mon→Hom-mon x mon) (_∘ f)
  precompose-hom-mon-hom {mon = mon} f .pres-id =
    (mon .η ∘ !) ∘ f ≡⟨ pullr (sym (!-unique (! ∘ f))) ⟩
    mon .η ∘ !       ∎
  precompose-hom-mon-hom {mon = mon} f .pres-⋆ g h =
    (mon .μ ∘ ⟨ g , h ⟩) ∘ f   ≡⟨ pullr (⟨⟩∘ f) ⟩
    mon .μ ∘ ⟨ g ∘ f , h ∘ f ⟩ ∎
```

We've almost shown that a monoid object $M : \cC$ fits into a _presheaf
of monoids_, a functor $\cC\op \to \thecat{Mon}$, mapping objects of
$\cC$ to the monoid of generalised elements $X \to M$. All that remains
is to show functoriality, which follows immediately:

```agda
  Mon→PshMon
    : ∀ {m} → C-Monoid m
    → Functor (C ^op) (Monoids ℓ)
  Mon→PshMon {m} mon .F₀ x .fst = el! (Hom x m)
  Mon→PshMon {m} mon .F₀ x .snd = Mon→Hom-mon x mon

  Mon→PshMon {m} mon .F₁ f .hom       = _∘ f
  Mon→PshMon {m} mon .F₁ f .preserves = precompose-hom-mon-hom {mon = mon} f

  Mon→PshMon {m} mon .F-id    = ext idr
  Mon→PshMon {m} mon .F-∘ f g = ext λ h → assoc h g f
```

And, since this presheaf is _by definition_ given by the set of maps
into an object, it's representable!

```agda
  Mon→PshMon-rep
    : ∀ {m} → (mon : C-Monoid m)
    → Representation {C = C} (Mon↪Sets F∘ Mon→PshMon mon)
  Mon→PshMon-rep {m = m} mon .rep = m
  Mon→PshMon-rep {m = m} mon .represents = to-natural-iso ni where
    open make-natural-iso

    ni : make-natural-iso (Mon↪Sets F∘ Mon→PshMon mon) (Hom-into C m)
    ni .eta _ f   = f
    ni .inv _ f   = f
    ni .eta∘inv _ = refl
    ni .inv∘eta _ = refl
    ni .natural _ _ _ = refl
```

Now, suppose we have a pair of monoid objects, $M$ and $N$, together
with a homomorphism $f : M \to N$. We can now consider the
*post*composition with $f$, a function of sets which maps between the
relativizations of $M$ and $N$ to arbitrary contexts: it has type

$$
\hom(X, M) \to \hom(X, N)
$$.

Since we've equipped these sets with monoid structures using the
internal structures on $M$ and $N$, and $f$ is a homomorphism between
those, we would like for postcomposition with $f$ to _also_ be a monoid
homomorphism.... which it is!

```agda
  internal-mon-hom→hom-mon-hom
    : ∀ {x m n} {f : Hom m n} {m-mon : C-Monoid m} {n-mon : C-Monoid n}
    → C-Monoid-hom f m-mon n-mon
    → Monoid-hom (Mon→Hom-mon x m-mon) (Mon→Hom-mon x n-mon) (f ∘_)
  internal-mon-hom→hom-mon-hom {f = f} {m-mon} {n-mon} hom .pres-id =
    f ∘ m-mon .η ∘ ! ≡⟨ pulll (hom .pres-η) ⟩
    n-mon .η ∘ !     ∎
  internal-mon-hom→hom-mon-hom {f = f} {m-mon} {n-mon} hom .pres-⋆ g h =
    f ∘ m-mon .μ ∘ ⟨ g , h ⟩       ≡⟨ extendl (hom .pres-μ) ⟩
    n-mon .μ ∘ f ⊗₁ f ∘ ⟨ g , h ⟩  ≡⟨ products! prod ⟩
    n-mon .μ ∘ ⟨ f ∘ g , f ∘ h ⟩   ∎
```

To recap, these are the facts:

- A monoid object $M$ _externalises_ to a family of $\Sets$-monoids
$\hom(X, M)$, where $X : \cC$ is an arbitrary object we affectionately
refer to as the "context".

- Maps $f : X \to Y$ act by precomposition, which, extending the
analogy, corresponds to _substitution_. Substitution along arbitrary
maps is a monoid homomorphism, so $\yo(M)$ extends to a functor $\cC\op
\to \thecat{Mon}$, a **representable presheaf of monoids**;

- Monoid homomorphisms $f : M \to N$, when acting by postcomposition,
externalise to $\Sets$-monoid homomorphisms $\hom(X, M) \to \hom(X, N)$.

<!--
```agda
  private
    Mon[C] : Precategory (o ⊔ ℓ) (ℓ ⊔ ℓ)
    Mon[C] = ∫ Mon[ Cartesian-monoidal prod term ]

  PShMon : ∀ κ → Precategory (o ⊔ ℓ ⊔ lsuc κ) (o ⊔ ℓ ⊔ κ)
  PShMon κ = Cat[ C ^op , Monoids κ ]

  RepPShMon : Precategory (o ⊔ lsuc ℓ) (o ⊔ ℓ)
  RepPShMon = Restrict {C = PShMon ℓ} (λ P → Representation {C = C} (Mon↪Sets F∘ P))
```
-->

To make this correspondence formal, we'll define the category of
**representable presheaves of monoids** to be the full subcategory of
$\cC\op \to \thecat{Mon}$ on the representable objects; for now, it will
be denoted $\psh(\cC)_M$ --- a notation for which the author apologises.
As usual, $\thecat{Mon}_\cC$ will denote the category of monoid objects
on $\cC$.

We have described most of a functor $\thecat{Mon}_\cC \to \psh(\cC)_M$.
It only remains to verify that the action by postcomposition of a monoid
homomorphism $f : M \to N$ is a natural transformation $\hom(-, M) \to
\hom(-, N)$.

```agda
  Mon→RepPShMon : Functor Mon[C] RepPShMon
  Mon→RepPShMon .F₀ (m , mon) .fst = Mon→PshMon mon
  Mon→RepPShMon .F₀ (m , mon) .snd = Mon→PshMon-rep mon

  Mon→RepPShMon .F₁ f .η x .hom = f .hom ∘_
  Mon→RepPShMon .F₁ f .η x .preserves =
    internal-mon-hom→hom-mon-hom (f .preserves)
  Mon→RepPShMon .F₁ f .is-natural x y g = ext λ h → assoc (f .hom) h g

  Mon→RepPShMon .F-id = ext λ x f → idl f
  Mon→RepPShMon .F-∘ f g = ext λ x h → sym (assoc (f .hom) (g .hom) h)
```

This functor is a simultaneous restriction and corestriction of the
Yoneda embedding on $\cC$. After calculating that natural
transformations between representable presheaves of monoids determine
monoid homomorphisms^[Evaluating their components at the identity
morphism, as usual!], the usual argument will suffice to show that this
functor is also [[fully faithful]].

```agda
  Nat→internal-mon-hom
    : ∀ {m n} {m-mon : C-Monoid m} {n-mon : C-Monoid n}
    → (α : Mon→PshMon m-mon => Mon→PshMon n-mon)
    → C-Monoid-hom (α .η m # id) m-mon n-mon
  Nat→internal-mon-hom {m} {n} {m-mon} {n-mon} α .pres-η =
    (α .η m # id) ∘ (m-mon .η) ≡˘⟨ ap hom (α .is-natural _ _ _) $ₚ _ ⟩
    α .η top # (id ∘ m-mon .η) ≡⟨ ap (α .η _ #_) (id-comm-sym ∙ ap (m-mon .η ∘_) (sym (!-unique _))) ⟩
    α .η top # (m-mon .η ∘ !)  ≡⟨ α .η _ .preserves .pres-id ⟩
    n-mon .η ∘ !               ≡⟨ elimr (!-unique _) ⟩
    n-mon .η                   ∎
  Nat→internal-mon-hom {m} {n} {m-mon} {n-mon} α .pres-μ =
    α .η m # id ∘ (m-mon .μ)                               ≡˘⟨ ap hom (α .is-natural _ _ _) $ₚ _ ⟩
    α .η (m ⊗₀ m) # (id ∘ m-mon .μ)                        ≡⟨ ap (α .η _ #_) (id-comm-sym ∙ ap (m-mon .μ ∘_) (sym ⟨⟩-η)) ⟩
    α .η (m ⊗₀ m) # (m-mon .μ ∘ ⟨ π₁ , π₂ ⟩)               ≡⟨ α .η _ .preserves .pres-⋆ _ _ ⟩
    n-mon .μ ∘ ⟨ α .η _ # π₁ , α .η _ # π₂ ⟩               ≡˘⟨ ap (n-mon .μ ∘_) (ap₂ ⟨_,_⟩ (ap (α .η _ #_) (idl _)) (ap (α .η _ #_) (idl _))) ⟩
    n-mon .μ ∘ ⟨ α .η _ # (id ∘ π₁) , α .η _ # (id ∘ π₂) ⟩ ≡⟨ ap (n-mon .μ ∘_) (ap₂ ⟨_,_⟩ (ap hom (α .is-natural _ _ _) $ₚ _) (ap hom (α .is-natural _ _ _) $ₚ _)) ⟩
    n-mon .μ ∘ (α .η m # id ⊗₁ α .η m # id)                ∎

  open is-iso

  Mon→RepPShMon-is-ff : is-fully-faithful Mon→RepPShMon
  Mon→RepPShMon-is-ff = is-iso→is-equiv λ where
    .inv α .hom       → α .η _ # id
    .inv α .preserves → Nat→internal-mon-hom α
    .rinv α → ext λ _ f →
      α .η _ # id ∘ f   ≡˘⟨ ap hom (α .is-natural _ _ _) $ₚ _ ⟩
      α .η _ # (id ∘ f) ≡⟨ ap (α .η _ #_) (idl f) ⟩
      α .η _ # f        ∎
    .linv h → total-hom-path _ (idr _) prop!
```

# Internalizing presheaves of monoids

Intuitively, what we have just shown is that monoids internal to $\cC$
yield monoids in the internal language of $\cC$ --- giving monoids in
_arbitrary_ contexts, in a manner compatible with substitution. We will
now establish the converse: If $\hom(-, M)$ is always a monoid, then $M$
is a monoid object, as long as the monoid structures are stable under
substitution --- dropping the analogy, as long as the monoid structure
is natural.

<!--
```agda
  module _ {m : Ob} (mon : ∀ x → Monoid-on (Hom x m)) where
    private module mon {x} = Monoid-on (mon x)
    open mon using (identity; _⋆_)
```
-->

```agda
    Hom-mon→Mon
      : (∀ {x y} (f : Hom x y) → identity ∘ f ≡ identity)
      → (∀ {x y} (f g : Hom y m) (h : Hom x y) → (f ⋆ g) ∘ h ≡ f ∘ h ⋆ g ∘ h)
      → C-Monoid m
```

The monoid operations are defined in the smallest context possible. For
identity this is the empty context^[The [[terminal object]].], for
multiplication, this is the context $M \times M$.

```agda
    Hom-mon→Mon id-nat ⋆-nat .η = identity {top}
    Hom-mon→Mon id-nat ⋆-nat .μ = π₁ ⋆ π₂
```

To establish the monoid laws, we'll use the naturality conditions we
imposed on the monoids $\hom(-, M)$.

```agda
    Hom-mon→Mon id-nat ⋆-nat .μ-unitl =
      (π₁ ⋆ π₂) ∘ (identity ⊗₁ id)                    ≡⟨ ⋆-nat _ _ _ ⟩
      (π₁ ∘ identity ⊗₁ id) ⋆ (π₂ ∘ identity ⊗₁ id)   ≡⟨ ap₂ _⋆_ π₁∘⟨⟩ π₂∘⟨⟩ ⟩
      (identity ∘ π₁) ⋆ (id ∘ π₂)                     ≡⟨ ap₂ _⋆_ (id-nat _) (idl _) ⟩
      identity ⋆ π₂                                   ≡⟨ mon.idl ⟩
      π₂                                              ∎

    Hom-mon→Mon id-nat ⋆-nat .μ-unitr =
      (π₁ ⋆ π₂) ∘ (id ⊗₁ identity)                  ≡⟨ ⋆-nat _ _ _ ⟩
      (π₁ ∘ id ⊗₁ identity) ⋆ (π₂ ∘ id ⊗₁ identity) ≡⟨ ap₂ _⋆_ π₁∘⟨⟩ π₂∘⟨⟩ ⟩
      (id ∘ π₁) ⋆ (identity ∘ π₂)                   ≡⟨ ap₂ _⋆_ (idl _) (id-nat _) ⟩
      π₁ ⋆ identity                                 ≡⟨ mon.idr ⟩
      π₁                                            ∎

    Hom-mon→Mon id-nat ⋆-nat .μ-assoc =
      (π₁ ⋆ π₂) ∘ (id ⊗₁ (π₁ ⋆ π₂))                                  ≡⟨ ⋆-nat _ _ _ ⟩
      (π₁ ∘ id ⊗₁ (π₁ ⋆ π₂)) ⋆ (π₂ ∘ id ⊗₁ (π₁ ⋆ π₂))                ≡⟨ ap₂ _⋆_ π₁∘⟨⟩ π₂∘⟨⟩ ⟩
      (id ∘ π₁) ⋆ ((π₁ ⋆ π₂) ∘ π₂)                                   ≡⟨ ap₂ _⋆_ (idl _) (⋆-nat _ _ _) ⟩
      π₁ ⋆ ((π₁ ∘ π₂) ⋆ (π₂ ∘ π₂))                                   ≡⟨ mon.associative ⟩
      (π₁ ⋆ (π₁ ∘ π₂)) ⋆ (π₂ ∘ π₂)                                   ≡˘⟨ ap₂ _⋆_ (⋆-nat _ _ _ ∙ ap₂ _⋆_ π₁∘⟨⟩ π₂∘⟨⟩) (idl _) ⟩
      ((π₁ ⋆ π₂) ∘ ⟨ π₁ , π₁ ∘ π₂ ⟩) ⋆ (id ∘ π₂ ∘ π₂)                ≡⟨ ap₂ _⋆_ (ap ((π₁ ⋆ π₂) ∘_) (sym π₁∘⟨⟩)) (ap (id ∘_) (sym π₂∘⟨⟩)) ⟩
      ((π₁ ⋆ π₂) ∘ π₁ ∘ _) ⋆ (id ∘ π₂ ∘ _)                           ≡⟨ ap₂ _⋆_ (extendl (sym π₁∘⟨⟩)) (extendl (sym π₂∘⟨⟩)) ⟩
      (π₁ ∘ _) ⋆ (π₂ ∘ _)                                            ≡˘⟨ ⋆-nat _ _ _ ⟩
      (π₁ ⋆ π₂) ∘ ((π₁ ⋆ π₂) ⊗₁ id) ∘ ⟨ ⟨ π₁ , π₁ ∘ π₂ ⟩ , π₂ ∘ π₂ ⟩ ∎
```

We will use this construction to construct the inverse of our
externalisation functor. If we have a representable presheaf of monoids
$P$, then, by definition, we have substitution-stable monoid structures
on $P(-)$, and natural isomorphisms $P(-) \cong \hom(-, M)$, for some
object $M : \cC$.

```agda
  RepPshMon→Mon
    : ∀ (P : Functor (C ^op) (Monoids ℓ))
    → (P-rep : Representation {C = C} (Mon↪Sets F∘ P))
    → C-Monoid (P-rep .rep)
  RepPshMon→Mon P P-rep = Hom-mon→Mon hom-mon η*-nat μ*-nat
    module RepPshMon→Mon where
```

<!--
```agda
    m : Ob
    m = P-rep .rep

    PMon : Ob → Type ℓ
    PMon x = ∣ P .F₀ x .fst ∣

    module PMon {x} = Monoid-on (P .F₀ x .snd)
    module repr = Isoⁿ (P-rep .represents)

    open PMon hiding (idl; idr; associative)

    gen : ∀ {x} → PMon x → Hom x m
    gen {x} px = repr.to .η x px

    elt : ∀ {x} → Hom x m → PMon x
    elt {x} f = repr.from .η x f
```
-->

As noted, the representability condition means we have specified
isomorphisms between the sets $P(x)$ --- the _sections_ of $P$ --- and
generalised objects $x \to M$, where $M$ is the representing object. It
follows, _even if this isomorphism is not natural_, that we can transfer
the monoid structure $P(x)$ to a monoid structure on $x \to M$.

```agda
    η* : ∀ x → Hom x m
    η* x = gen identity

    μ* : ∀ {x} → Hom x m → Hom x m → Hom x m
    μ* {x = x} f g = gen $ (elt f) ⋆ (elt g)
```

<details>
<summary>There is no surprise to the calculation establishing the monoid laws, here.</summary>

```agda
    η*-idl : ∀ {x} → (f : Hom x m) → μ* (η* x) f ≡ f
    η*-idl {x} f =
      gen (⌜ elt (gen identity) ⌝ ⋆ elt f) ≡⟨ ap! (unext repr.invr _ _) ⟩
      gen (identity ⋆ elt f)               ≡⟨ ap gen PMon.idl ⟩
      gen (elt f)                          ≡⟨ unext repr.invl _ _ ⟩
      f                                    ∎

    η*-idr : ∀ {x} → (f : Hom x m) → μ* f (η* x) ≡ f
    η*-idr {x} f =
      gen (elt f ⋆ ⌜ elt (gen identity) ⌝) ≡⟨ ap! (unext repr.invr _ _) ⟩
      gen (elt f ⋆ identity)               ≡⟨ ap gen PMon.idr ⟩
      gen (elt f)                          ≡⟨ unext repr.invl _ _ ⟩
      f                                    ∎

    μ*-assoc : ∀ {x} → (f g h : Hom x m) → μ* f (μ* g h) ≡ μ* (μ* f g) h
    μ*-assoc {x} f g h =
      gen (elt f ⋆ ⌜ elt (gen (elt g ⋆ elt h)) ⌝) ≡⟨ ap! (unext repr.invr _ _) ⟩
      gen (elt f ⋆ (elt g ⋆ elt h))               ≡⟨ ap gen PMon.associative ⟩
      gen (⌜ elt f ⋆ elt g ⌝ ⋆ elt h)             ≡⟨ ap! (sym $ unext repr.invr _ _) ⟩
      gen (elt (gen (elt f ⋆ elt g)) ⋆ elt h)     ∎
```
</details>

<!--
```agda
    hom-mon : ∀ x → Monoid-on (Hom x m)
    hom-mon x .Monoid-on.identity = η* x
    hom-mon x .Monoid-on._⋆_ = μ*
    hom-mon x .Monoid-on.has-is-monoid .has-is-semigroup .has-is-magma .has-is-set =
      Hom-set x m
    hom-mon x .Monoid-on.has-is-monoid .has-is-semigroup .associative = μ*-assoc _ _ _
    hom-mon x .Monoid-on.has-is-monoid .mon-idl = η*-idl _
    hom-mon x .Monoid-on.has-is-monoid .mon-idr = η*-idr _
```
-->

It remains to show that this assignment is natural --- which is why we
asked for a _natural_ isomorphism! A calculation mildly annoying
establishes the stability of identity and multiplication under
substitution.

```agda
    η*-nat
      : ∀ {w x} (f : Hom w x)
      → η* x ∘ f ≡ η* w
    η*-nat {w} {x} f =
      (η* x) ∘ f                  ≡˘⟨ repr.to .is-natural _ _ _ $ₚ _ ⟩
      gen (P .F₁ f .hom identity) ≡⟨ ap gen (P .F₁ f .preserves .pres-id) ⟩
      η* w ∎

    μ*-nat
      : ∀ {w x} (f g : Hom x m) (h : Hom w x)
      → μ* f g ∘ h ≡ μ* (f ∘ h) (g ∘ h)
    μ*-nat f g h =
      μ* f g ∘ h                                            ≡˘⟨ repr.to .is-natural _ _ _ $ₚ _ ⟩
      gen (P .F₁ h .hom ((elt f) ⋆ (elt g)))                ≡⟨ ap gen (P .F₁ h .preserves .pres-⋆ _ _) ⟩
      gen ((P .F₁ h .hom (elt f)) ⋆ (P .F₁ h .hom (elt g))) ≡˘⟨ ap gen (ap₂ _⋆_ (repr.from .is-natural _ _ _ $ₚ _) (repr.from .is-natural _ _ _ $ₚ _)) ⟩
      μ* (f ∘ h) (g ∘ h) ∎
```

We now have a construction mapping representable presheaves to monoid
objects. The last bit of algebra in this module establishes that
internalisation followed by externalisation produces a presheaf of
monoids isomorphic to the one we started with.

```agda
  Mon→RepPShMon-is-split-eso : is-split-eso Mon→RepPShMon
  Mon→RepPShMon-is-split-eso (P , pm) .fst = pm .rep , RepPshMon→Mon P pm
  Mon→RepPShMon-is-split-eso (P , pm) .snd = super-iso→sub-iso _ $ to-natural-iso ni where
    open make-natural-iso
    open RepPshMon→Mon P pm
    open PMon using (identity; _⋆_)
    module P = Functor P
```

<details>
<summary>If you still have the patience for some more algebra, you can
expand this `<details>` element.</summary>

```agda
    ni : make-natural-iso (Mon→PshMon (RepPshMon→Mon P pm)) P
    ni .eta x .hom = repr.from .η x
    ni .inv x .hom = repr.to .η x

    ni .eta x .preserves .pres-id =
      elt (η* top ∘ !)           ≡⟨ ap elt (η*-nat !) ⟩
      elt (η* x)                 ≡⟨ unext repr.invr _ _ ⟩
      identity                   ∎
    ni .eta x .preserves .pres-⋆ f g =
      elt (μ* π₁ π₂ ∘ ⟨ f , g ⟩)                 ≡⟨ ap elt (μ*-nat _ _ _) ⟩
      elt (μ* (π₁ ∘ ⟨ f , g ⟩) (π₂ ∘ ⟨ f , g ⟩)) ≡⟨ ap elt (ap₂ μ* π₁∘⟨⟩ π₂∘⟨⟩) ⟩
      elt (μ* f g)                               ≡⟨ unext repr.invr _ _ ⟩
      (elt f ⋆ elt g)                            ∎

    ni .inv x .preserves .pres-id = sym (η*-nat _)
    ni .inv x .preserves .pres-⋆ f g =
      gen (f ⋆ g)                                          ≡˘⟨ ap gen (ap₂ _⋆_ (unext repr.invr _ _) (unext repr.invr _ _)) ⟩
      μ* (gen f) (gen g)                                   ≡˘⟨ ap₂ μ* π₁∘⟨⟩ π₂∘⟨⟩ ⟩
      μ* (π₁ ∘ ⟨ gen f , gen g ⟩) (π₂ ∘ ⟨ gen f , gen g ⟩) ≡˘⟨ μ*-nat _ _ _ ⟩
      μ* π₁ π₂ ∘ ⟨ gen f , gen g ⟩                         ∎

    ni .eta∘inv x = ext (unext repr.invr x)
    ni .inv∘eta x = ext (unext repr.invl x)
    ni .natural x y f = ext (sym (repr.from .is-natural _ _ _) $ₚ_)
```
</details>

Put another way, our functor $\thecat{Mon}_\cC \to \psh(\cC)_M$ is a
[split essential surjection] --- so, remembering that it was fully
faithful, we conclude it's an equivalence.

[split essential surjection]: Cat.Functor.Properties.html#essential-fibres

```agda
  Mon→RepPShMon-is-equiv : is-equivalence Mon→RepPShMon
  Mon→RepPShMon-is-equiv = ff+split-eso→is-equivalence
    Mon→RepPShMon-is-ff
    Mon→RepPShMon-is-split-eso
```

## The big picture

It's easy to lose the forest for the trees here, so let's take a step
back and examine what we have done. This equivalence of categories shows
that monoids in the internal language of $\cC$ are really monoids in
$\cC$. Furthermore, nothing we have done relies on the structure of
monoids; we could repeat the same argument with internal groups and
everything would go through smoothly.

The lesson is that category theorists prefer to define internal structure
in the smallest context possible, and then rely on weakening to obtain
a well-behaved object in the internal language. This *works*, but is
somewhat unnatural, and can summon pullback nasal demons that will ruin
your day. For instance, defining internal categories in this manner
requires taking pullbacks to ensure that the composition operation is
well formed, which spirals out of control when quantifying over multiple
morphisms due to coherence issues. If we take this generalized
object perspective instead, such coherence issues can be avoided!
