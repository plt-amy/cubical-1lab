<!--
```agda
open import Cat.Diagram.Monad.Relative
open import Cat.Instances.Functor
open import Cat.Diagram.Monad
open import Cat.Prelude

import Cat.Reasoning

open Functor
open _=>_
```
-->

```agda
module Cat.Diagram.Monad.Extension where
```

<!--
```agda
module _ {o ℓ} (C : Precategory o ℓ) where
```
-->

# Extension systems {defines=extension-system}

An **extension system** on a category $\cC$ is an alternative way of
presenting a [[monad]] on $\cC$. As the name suggests, extension systems
are built around an *extension* operation, of type $\cC(X,MY) \to
\cC(MX, MY)$, in place of the *multiplication* operation $\cC(MMX, MX)$
that characterises a monad. This has a couple of immediate benefits:

1. The extension operation avoids nested applications of $M$, which
   tends to improve ergonomics.

2. The extension operation $\cC(X,MY) \to \cC(MX, MY)$ simultaneously
   encodes both $M$'s multiplication *and* its underlying functorial
   action, which reduces the amount of data that needs to be given
   explicitly.

3. It is not immediately clear how to generalize monads beyond
   endofunctors.
   In constrast, extension systems can be readily generalized to
   [[relative extension systems]]^[In fact, we will *define* extension
   systems as a special case of relative extension systems!].

With that bit of motivation out of the way, we shall proceed to define
extension systems. An extension system consists of:

1. A mapping of objects, $M : \cC \to \cC$; and

2. A family of morphisms $\eta_X : \cC(X, MX)$, called the **unit**
   of the extension system; and

3. The extension operation, $(-)^{M} : \cC(X,MY) \to \cC(MX, MY)$.
Gesturing towards the "monads" found in functional programming, we will
call this operation `bind`.

Note that we do not require the mapping of objects to be functorial, nor
the do we require the unit to be natural. Instead, we impose 3 equations
on this structure:^[contrast this with the 7 equations required for a
monad: 2 for functoriality, 2 for naturality, and 3 for
unitality/associativity]

1. For every $X : \cC$, we must have $(\eta_X)^M = \id_{MX}$;
2. For every $f : \cC(X, MY)$, we must have $f^M \circ \eta_X = f$; and
3. For every $f : \cC(Y, MX)$, and $g : \cC(X, MY)$, we must have
   $f^M \circ g^M = (f^M \circ g)^M$.

For reasons of generality, we shall define extension systems as
[[relative extension systems]], along the identity functor. Even though
we use a more general definition, the data contained in an extension
system is precisely the data listed above.

```agda
Extension-system : ∀ {o ℓ} → Precategory o ℓ → Type (o ⊔ ℓ)
Extension-system C = Relative-extension-system (Id {C = C})

module Extension-system {o ℓ} {C : Precategory o ℓ} (E : Extension-system C) where
  open Cat.Reasoning C
  open Relative-extension-system E public
```

We can recover the monad's multiplication by extending the identity
morphism $\id_{MX} : \cC(MX, MX)$.

```agda
  join : ∀ {x} → Hom (M₀ (M₀ x)) (M₀ x)
  join {x} = bind (id {M₀ x})
```

Naturality of join also follows, though is a bit more involved.

```agda
  join-natural
    : ∀ {x y} (f : Hom x y)
    → join ∘ M₁ (M₁ f) ≡ M₁ f ∘ join
  join-natural f =
    bind id ∘ bind (unit ∘ bind (unit ∘ f)) ≡⟨ bind-∘ id (unit ∘ bind (unit ∘ f)) ⟩
    bind (bind id ∘ unit ∘ bind (unit ∘ f)) ≡⟨ ap bind (cancell (bind-unit-∘ id) ∙ sym (idr _)) ⟩
    bind (bind (unit ∘ f) ∘ id)             ≡˘⟨ bind-∘ (unit ∘ f) id ⟩
    bind (unit ∘ f) ∘ bind id               ∎
```

## Equivalence with monads

<!--
```agda
module _ {o ℓ} {C : Precategory o ℓ} where
  open Cat.Reasoning C
```
-->

It remains to show that monads on $\cC$ are equivalent to extension
systems on $\cC$. We'll start with the forward direction. Throughout,
let $M$ be a fixed monad on $\cC$.

```agda
  Monad→Extension-system : Monad C → Extension-system C
  Monad→Extension-system (_ , M) = system where
    module M = Monad-on M
    open Extension-system
```

The mapping of objects, and the unit, are directly given by (the object
part of) $M$-qua-endofunctor, and by the unit natural transformation.

```agda
    system : Extension-system C
    system .M₀ = M.M₀
    system .unit = M.η _
```

Defining the extension operation is slightly trickier, but not by much.
If we have a morphism $f : \cC(X, MY)$, then its extension is defined to
be composite

$$
MX \xto{Mf} MMY \xto{\mu} MY
$$.

```agda
    system .bind f = M.μ _ ∘ M.M₁ f
```

Finally, a few short computations show that this definition is lawful.

```agda
    system .bind-unit-id =
      M.μ _ ∘ M.M₁ (M.η _) ≡⟨ M.μ-idl ⟩
      id                             ∎
    system .bind-unit-∘ f =
      (M.μ _ ∘ M.M₁ f) ∘ M.η _ ≡⟨ pullr (sym $ M.unit.is-natural _ _ _) ⟩
      M.μ _ ∘ M.η _ ∘ f        ≡⟨ cancell M.μ-idr ⟩
      f                        ∎
    system .bind-∘ f g =
      (M.μ _ ∘ M.M₁ f) ∘ (M.μ _ ∘ M.M₁ g)             ≡⟨ pullr (extendl (sym $ M.mult.is-natural _ _ _)) ⟩
      M.μ _ ∘ M.μ _ ∘ (M.M₁ (M.M₁ f) ∘ M.M₁ g)        ≡⟨ extendl (sym M.μ-assoc) ⟩
      M.μ _ ∘ M.M₁ (M.μ _) ∘ (M.M₁ (M.M₁ f) ∘ M.M₁ g) ≡⟨ ap₂ _∘_ refl (pulll (sym (M.M-∘ _ _)) ∙ sym (M.M-∘ _ _)) ⟩
      M.μ _ ∘ M.M₁ ((M.μ _ ∘ M.M₁ f) ∘ g)             ∎
```

Constructing a monad from an extension system is simply a matter of
bolting together our results from the previous section.

```agda
  Extension-system→Monad : Extension-system C → Monad C
  Extension-system→Monad E = _ , monad where
    module E = Extension-system E
    open Monad-on

    monad : Monad-on E.M
    monad .unit .η x = E.unit
    monad .unit .is-natural _ _ f = E.unit-natural f
    monad .mult .η x = E.join
    monad .mult .is-natural _ _ f = E.join-natural f
```

The monad laws follow from another short series of computations.

```agda
    monad .μ-idl =
      E.bind id ∘ E.bind (E.unit ∘ E.unit) ≡⟨ E.bind-∘ _ _ ⟩
      E.bind (E.bind id ∘ E.unit ∘ E.unit) ≡⟨ ap E.bind (cancell (E.bind-unit-∘ id)) ⟩
      E.bind E.unit                        ≡⟨ E.bind-unit-id ⟩
      id                                   ∎
    monad .μ-idr =
      E.bind id ∘ E.unit ≡⟨ E.bind-unit-∘ id ⟩
      id                 ∎
    monad .μ-assoc =
      E.bind id ∘ E.bind (E.unit ∘ E.bind id) ≡⟨ E.bind-∘ _ _ ⟩
      E.bind (E.bind id ∘ E.unit ∘ E.bind id) ≡⟨ ap E.bind (cancell (E.bind-unit-∘ id) ∙ sym (idr _)) ⟩
      E.bind (E.bind id ∘ id)                 ≡˘⟨ E.bind-∘ _ _ ⟩
      E.bind id ∘ E.bind id                   ∎
```

<!--
```agda
  Extension-system→Monad-on : (E : Extension-system C) → Monad-on (Extension-system.M E)
  Extension-system→Monad-on E = Extension-system→Monad E .snd
```
-->

Moreover, these two functions constitute an equivalence between monads
on $\cC$ and extension systems on $\cC$. In light of this fact, we will
silently identify monads and extension systems, whenever it is
convenient.

```agda
  Monad≃Extension-system : Monad C ≃ Extension-system C
  Monad≃Extension-system = Iso→Equiv $
    Monad→Extension-system ,
    iso Extension-system→Monad
      (λ E →
        let module E = Extension-system E in
        Relative-extension-system-path
          (λ _ → refl)
          (λ _ → refl)
          (λ f →
            E.bind id ∘ E.bind (E.unit ∘ f i0) ≡⟨ E.bind-∘ id (E.unit ∘ f i0) ⟩
            E.bind (E.bind id ∘ E.unit ∘ f i0) ≡⟨ ap E.bind (cancell (E.bind-unit-∘ id)) ⟩
            E.bind (f i0)                      ≡⟨ ap E.bind (coe0→1 (λ i → f i0 ≡ f i) refl) ⟩
            E.bind (f i1)                      ∎))
      (λ M →
        let module M = Monad-on (M .snd) in
        Σ-pathp
          (Functor-path (λ _ → refl) (λ f →
            M.μ _ ∘ M.M₁ (M.η _ ∘ f)        ≡⟨ pushr (M.M-∘ _ _) ⟩
            (M.μ _ ∘ M.M₁ (M.η _)) ∘ M.M₁ f ≡⟨ eliml M.μ-idl ⟩
            M.M₁ f ∎))
          (Monad-on-path _ (λ _ → refl) (λ _ → elimr M.M-id)))
```

# Algebras over an extension system {defines=extension-algebra}

An **extension algebra** over $E$ is the extension system analog of a
[[algebra over a monad]]. Following the general theme of extension
operations, an extension algebra on $X : \cC$ is given by an operation
$\nu : \cC(A, X) \to \cC(MA, X)$. Intuitively, this operation lets us
"evaluate" any $M$, so long as the codomain of the evaluation is $X$.

This operation must satisfy a pair of equations:

1. For every $f : \cC(A, X)$, we have $\nu(f) \circ \eta_{A} = f$; and
2. For every $f : \cC(B, X)$, and $g : \cC(A, MB)$, we have $\nu(f) \circ g^M = \nu(\nu f \circ g)$.

As with extension systems, we define extension algebras in terms of
[[relative extension algebras]].

```agda
Extension-algebra-on
  : ∀ {o ℓ} {C : Precategory o ℓ}
  → (open Precategory C)
  → Extension-system C
  → Ob
  → Type (o ⊔ ℓ)
Extension-algebra-on = Relative-algebra-on

module Extension-algebra-on
  {o ℓ} {C : Precategory o ℓ} (open Cat.Reasoning C)
  {E : Extension-system C} {x : Ob} (α : Extension-algebra-on E x)
  where
  open Extension-system E
  open Relative-algebra-on α public
```

The evaluation map also interacts well with the multiplication.

```agda
  ν-join : ∀ {a} (f : Hom a x) → ν f ∘ join ≡ ν (ν f)
  ν-join f =
    ν f ∘ bind id ≡⟨ ν-bind f id ⟩
    ν (ν f ∘ id)  ≡⟨ ap ν (idr _) ⟩
    ν (ν f)       ∎
```

## Equivalence with monad algebras

<!--
```agda
module _ {o ℓ} {C : Precategory o ℓ} {E : Extension-system C} where
  open Cat.Reasoning C
  open Extension-system E
  open Extension-algebra-on

  private EM = Extension-system→Monad-on E
```
-->

As the name suggests, extension algebras over $E$ are equivalent to
monad algebras over the canonical monad associated with $E$.

For the forward direction, let $\alpha : MX \to X$ be a monad algebra
over $E$. We can obtain the required extension operation $\nu$ by
sending each $f : \cC(A, X)$ to the composite

$$
MA \xto{Mf} MX \xto{\alpha} X
$$.

```agda
  Algebra-on→Extension-algebra-on
    : ∀ {x} → Algebra-on EM x → Extension-algebra-on E x
  Algebra-on→Extension-algebra-on {x = x} α = ext-alg where
    module α = Algebra-on α
    open Extension-algebra-on

    ext-alg : Extension-algebra-on E x
    ext-alg .ν f = α.ν ∘ M₁ f
```

<details>
<summary>The monad algebra laws follow from some tedious calculations
that we shall omit.
</summary>

```agda
    ext-alg .ν-unit f =
      (α.ν ∘ M₁ f) ∘ unit ≡⟨ pullr (sym $ unit-natural f) ⟩
      α.ν ∘ unit ∘ f      ≡⟨ cancell α.ν-unit ⟩
      f ∎
    ext-alg .ν-bind f g =
      (α.ν ∘ M₁ f) ∘ bind g                    ≡⟨ pullr (bind-naturall _ _) ⟩
      α.ν ∘ bind ⌜ M₁ f ∘ g ⌝                  ≡⟨ ap! (insertl (bind-unit-∘ id)) ⟩
      α.ν ∘ bind (join ∘ unit ∘ M₁ f ∘ g)      ≡⟨ pushr (sym (bind-∘ _ _)) ⟩
      (α.ν ∘ join) ∘ bind (unit ∘ M₁ f ∘ g)    ≡⟨ pushl (sym $ α.ν-mult) ⟩
      α.ν ∘ M₁ α.ν ∘ bind (unit ∘ M₁ f ∘ g)    ≡⟨ ap (α.ν ∘_) (bind-naturall _ _) ⟩
      α.ν ∘ bind ⌜ M₁ α.ν ∘ unit ∘ M₁ f ∘ g ⌝  ≡⟨ ap! (centralizel (sym $ unit-natural _)) ⟩
      α.ν ∘ bind (unit ∘ (α.ν ∘ M₁ f) ∘ g)     ∎
```
</details>

Conversely, a monad algebra over $E$ can be derived from an extension
algebra $\nu : \cC(A, X) \to \cC(MA, X)$ over $E$ by restricting to
$\nu(\id_{X}) : \cC(MX, X)$.

```agda
  Extension-algebra-on→Algebra-on
    : ∀ {x} → Extension-algebra-on E x → Algebra-on EM x
  Extension-algebra-on→Algebra-on {x = x} α = alg where
    module α = Extension-algebra-on α
    open Algebra-on

    alg : Algebra-on (Extension-system→Monad E .snd) x
    alg .ν = α.ν id
```

The proofs of the monad algebra laws are mercifully short.

```agda
    alg .ν-unit = α.ν-unit id
    alg .ν-mult =
      α.ν id ∘ M₁ (α.ν id) ≡⟨ α.ν-natural _ _ ⟩
      α.ν (id ∘ α.ν id)    ≡⟨ ap α.ν (idl _) ⟩
      α.ν (α.ν id)         ≡˘⟨ α.ν-join id ⟩
      α.ν id ∘ join ∎
```

As expected, these two functions constitute an equivalence between monad
algebras and extension algebras.

```agda
  Algebra-on≃Extension-algebra-on
    : ∀ {x} → Algebra-on EM x ≃ Extension-algebra-on E x
  Algebra-on≃Extension-algebra-on {x} = Iso→Equiv $
    Algebra-on→Extension-algebra-on ,
    iso Extension-algebra-on→Algebra-on
      (λ α →
        let module α = Extension-algebra-on α in
        Relative-algebra-on-pathp refl λ f →
          α.ν id ∘ M₁ (f i0) ≡⟨ α.ν-natural _ _ ⟩
          α.ν (id ∘ f i0)    ≡⟨ ap α.ν (idl _) ⟩
          α.ν (f i0)         ≡⟨ ap α.ν (coe0→1 (λ i → f i0 ≡ f i) refl) ⟩
          α.ν (f i1)         ∎)
      (λ α →
        let module α = Algebra-on α in
        Algebra-on-pathp refl $
          α.ν ∘ bind (unit ∘ id) ≡⟨ elimr (ap bind (idr _) ∙ bind-unit-id) ⟩
          α.ν                    ∎)
```
