open import 1Lab.Reflection.Signature
open import 1Lab.Path.IdentitySystem
open import 1Lab.Function.Embedding
open import 1Lab.Reflection.HLevel
open import 1Lab.Reflection.Subst
open import 1Lab.HIT.Truncation
open import 1Lab.HLevel.Closure
open import 1Lab.Reflection
open import 1Lab.Type.Sigma
open import 1Lab.Type.Pi
open import 1Lab.HLevel
open import 1Lab.Equiv
open import 1Lab.Path
open import 1Lab.Type

open import Data.Nat.Base

module 1Lab.Extensionality where

{-
Automation for extensionality
=============================

The 'Extensional' typeclass equips a type with a “preferred” choice of
identity system. What “preferred” of course depends on the type under
consideration, but the vast majority of instances simply exist to note
that the projection of some record field is an embedding.

'Extensional' is quite a well-behaved typeclass. For starters, it is a
proposition: any pair of identity systems is connected by an equivalence
which preserves refl, and being an identity system is a proposition.

Of course, identity systems are not *definitionally* unique. However,
unlike classes which equip a type with *structure* (e.g., our very own
Meta.Append, or From-list, etc), a change in which Extensional instance
is selected can not change the *meaning* of a program: it can only
change whether or not the program actually elaborates.

Extensionality instances
------------------------

Every type has a default 'Extensional' instance, with the underlying
identity system being that of paths. Using instance overlap pragmas, we
can instruct Agda to only select the default instance in case it has
nothing else to try.

All other instances serve as "reduction rules". For example, extensional
equality for functions will, by default, be pointwise extensional
equality in the codomain; extensionality for pairs can also be
pointwise.

However, it's important that the "reduction rules" are maximally lazy.
This is because the 'Extensional' class is not actually definitionally
confluent. For example, we might expect that, since we have

  Extensionality-Π : ⦃ Extensional B ⦄ → Extensional (A → B)

then the instance for equivalences should be

  Extensionality-≃ : ⦃ Extensional B ⦄ → Extensional (A ≃ B)

but this is not actually the case: for specific instantiations of A and
B, it might be the case that a rule more specific than Extensional-Π can
fire (e.g. A is a quotient). The instance should instead be

  Extensionality-≃ : ⦃ Extensional (A → B) ⦄ → Extensional (A ≃ B)

which *only* applies the fact that is-equiv is a proposition, and does
not apply function extensionality.

Entry points
------------

While it would be possible to define a global relation _∼_ which
computes to the relation underlying a type's Extensional instance, this
would be pretty useless: the extensional instance would be frozen when
*the relation itself* is used, not when its values are used (or
introduced).

Our overarching philosophy is that Extensional computes "the domain of a
smart constructor for equality"; therefore, we only expose a few entry
points:

- ext:      turn extensional equality into equality
- unext:    the opposite
- reext!:   a macro which abbreviates "ext (unext p)"
- trivial!: a macro which abbreviates "ext (_ .reflᵉ _)"
-}

private variable
  ℓ ℓ' ℓ'' ℓr : Level
  A B C : Type ℓ
  P Q R : A → Type ℓ

record Extensional (A : Type ℓ) ℓ-rel : Type (ℓ ⊔ lsuc ℓ-rel) where
  no-eta-equality
  field
    Pathᵉ : A → A → Type ℓ-rel
    reflᵉ : ∀ x → Pathᵉ x x
    idsᵉ : is-identity-system Pathᵉ reflᵉ

open Extensional using (Pathᵉ ; reflᵉ ; idsᵉ) public

instance
  Extensional-default : Extensional A (level-of A)
  Extensional-default .Pathᵉ   = _≡_
  Extensional-default .reflᵉ _ = refl
  Extensional-default .idsᵉ    = Path-identity-system

  -- We can't mark this instance as OVERLAPPABLE because it's not
  -- strictly less specific than most other instances (it fixes the
  -- level of the relation to be that of the type).
  {-# INCOHERENT Extensional-default #-}

  -- Some vanilla "reduction rules": these all simply apply a
  -- pre-existing extensionality lemma. E.g., equality for values in a
  -- lifted type is equality of the underlying values, or equality of
  -- functions is pointwise.

  Extensional-Lift : ⦃ Extensional A ℓr ⦄ → Extensional (Lift ℓ' A) ℓr
  Extensional-Lift ⦃ sa ⦄ .Pathᵉ (lift x) (lift y) = sa .Pathᵉ x y
  Extensional-Lift ⦃ sa ⦄ .reflᵉ (lift x) = sa .reflᵉ x
  Extensional-Lift ⦃ sa ⦄ .idsᵉ .to-path p = ap lift (sa .idsᵉ .to-path p)
  Extensional-Lift ⦃ sa ⦄ .idsᵉ .to-path-over p = sa .idsᵉ .to-path-over p

  Extensional-Π
    : ⦃ ∀ {x} → Extensional (P x) ℓr ⦄
    → Extensional ((x : A) → P x) (level-of A ⊔ ℓr)
  Extensional-Π ⦃ sb ⦄ .Pathᵉ f g = ∀ x → Pathᵉ sb (f x) (g x)
  Extensional-Π ⦃ sb ⦄ .reflᵉ f x = reflᵉ sb (f x)
  Extensional-Π ⦃ sb ⦄ .idsᵉ .to-path h = funext λ i → sb .idsᵉ .to-path (h i)
  Extensional-Π ⦃ sb ⦄ .idsᵉ .to-path-over h = funextP λ i → sb .idsᵉ .to-path-over (h i)

  -- This instance is *very often* specialised.
  {-# OVERLAPPABLE Extensional-Π #-}

  Extensional-Π'
    : ⦃ ∀ {x} → Extensional (P x) ℓr ⦄
    → Extensional ({x : A} → P x) (level-of A ⊔ ℓr)
  Extensional-Π' ⦃ sb ⦄ .Pathᵉ f g = ∀ {x} → Pathᵉ (sb {x}) f g
  Extensional-Π' ⦃ sb ⦄ .reflᵉ f = reflᵉ sb f
  Extensional-Π' ⦃ sb ⦄ .idsᵉ .to-path h i = sb .idsᵉ .to-path h i
  Extensional-Π' ⦃ sb ⦄ .idsᵉ .to-path-over h i = sb .idsᵉ .to-path-over h i

  Extensional-Π''
    : ⦃ ∀ ⦃ x ⦄ → Extensional (P x) ℓr ⦄
    → Extensional (⦃ x : A ⦄ → P x) (level-of A ⊔ ℓr)
  Extensional-Π'' ⦃ sb ⦄ .Pathᵉ f g = ∀ ⦃ x ⦄ → Pathᵉ (sb ⦃ x ⦄) f g
  Extensional-Π'' ⦃ sb ⦄ .reflᵉ f = reflᵉ sb f
  Extensional-Π'' ⦃ sb ⦄ .idsᵉ .to-path h i = sb .idsᵉ .to-path h i
  Extensional-Π'' ⦃ sb ⦄ .idsᵉ .to-path-over h i = sb .idsᵉ .to-path-over h i

  -- Some non-confluent "reduction rules" for extensionality are those
  -- for functions from a type with a mapping-out property; here, we can
  -- immediately define instances for functions from Σ-types (equality
  -- is equality after currying) and for functions from lifts (equality
  -- is equality after lifting).
  --
  -- These overlap the Extensional-Π instance. To have them selected for
  -- e.g. equivalences ((Σ A B) ≃ C), the instance for equivalences
  -- *needs* to ask for Extensional (Σ A B → C) instead of Extensional
  -- C.

  Extensional-uncurry
    : ∀ {ℓ ℓ' ℓ'' ℓr} {A : Type ℓ} {B : A → Type ℓ'} {C : (x : A) → B x → Type ℓ''}
    → ⦃ sb : Extensional ((x : A) (y : B x) → C x y) ℓr ⦄
    → Extensional ((p : Σ A B) → C (p .fst) (p .snd)) ℓr
  Extensional-uncurry ⦃ sb ⦄ .Pathᵉ f g = sb .Pathᵉ (curry f) (curry g)
  Extensional-uncurry ⦃ sb ⦄ .reflᵉ f = sb .reflᵉ (curry f)
  Extensional-uncurry ⦃ sb = sb ⦄ .idsᵉ .to-path h i (a , b) = sb .idsᵉ .to-path h i a b
  Extensional-uncurry ⦃ sb = sb ⦄ .idsᵉ .to-path-over h = sb .idsᵉ .to-path-over h

  Extensional-lift-map
    : ∀ {ℓ ℓ' ℓ'' ℓr} {A : Type ℓ} {B : Lift ℓ' A → Type ℓ''}
    → ⦃ sa : Extensional ((x : A) → B (lift x)) ℓr ⦄
    → Extensional ((x : Lift ℓ' A) → B x) ℓr
  Extensional-lift-map ⦃ sa = sa ⦄ .Pathᵉ f g = sa .Pathᵉ (f ∘ lift) (g ∘ lift)
  Extensional-lift-map ⦃ sa = sa ⦄ .reflᵉ x = sa .reflᵉ (x ∘ lift)
  Extensional-lift-map ⦃ sa = sa ⦄ .idsᵉ .to-path h i (lift x) = sa .idsᵉ .to-path h i x
  Extensional-lift-map ⦃ sa = sa ⦄ .idsᵉ .to-path-over h = sa .idsᵉ  .to-path-over h

ext
  : ∀ {ℓ ℓr} {A : Type ℓ} {x y : A} ⦃ r : Extensional A ℓr ⦄
  → Pathᵉ r x y → x ≡ y
ext ⦃ r ⦄ p = r .idsᵉ .to-path p

{-
Using the extensionality tactic we can define a "more general refl",
where the terms being compared are not definitionally equal, but they
inhabit a type with a good identity system for which 'r x : R x y' type
checks.

The idea is to write a function wrapping

  ext : ⦃ r : Extensional A ℓ ⦄ (p : Pathᵉ r x y) → x ≡ y

so that it gives (reflᵉ r x) as the default argument for p. We can use a
tactic argument to accomplish this.
-}

private
  trivial-worker
    : ∀ {ℓ ℓr} {A : Type ℓ} (r : Extensional A ℓr)
    → (x y : A) → Term → TC ⊤
  trivial-worker r x y goal = try where
    error : ∀ {ℓ} {A : Type ℓ} → TC A

    -- We already have our r : Extensional A ℓr, and this is a macro, so
    -- we can just check that r .reflᵉ x : R x y. If that's the case
    -- then we can use that as the argument, otherwise we can give a
    -- slightly better error message than what Agda would yell.
    try : TC ⊤
    try = do
      `r ← wait-for-type =<< quoteTC r
      `x ← quoteTC x
      unify goal (it reflᵉ ##ₙ `r ##ₙ `x) <|> error

    error = do
      `x ← quoteTC x
      `y ← quoteTC y
      typeError
        [ "trivial! failed: the values\n  "
        , termErr `x
        , "\nand\n  "
        , termErr `y
        , "\nare not extensionally equal by refl.\n"
        ]

{-
trivial! serves to replace proofs like

  Nat-path λ x → funext λ y → Nat-path λ z → Homomorphism-path λ a → refl

since this is

  ext λ x y z a → refl

and that argument is precisely reflexivity for the computed identity
system which 'ext' is using. By the way, this example is totally made
up.
-}

opaque
  trivial!
    : ∀ {ℓ ℓr} {A : Type ℓ} {x y : A}
    → ⦃ r : Extensional A ℓr ⦄
    → {@(tactic trivial-worker r x y) p : Pathᵉ r x y}
    → x ≡ y
  trivial! ⦃ r ⦄ {p = p} = r .idsᵉ .to-path p

-- Helper for constructing isomorphisms where both equations hold
-- via 'trivial!'
trivial-iso!
  : ∀ {ℓ ℓ' ℓr ℓs} {A : Type ℓ} {B : Type ℓ'}
  → ⦃ r : Extensional (A → A) ℓr ⦄
  → ⦃ s : Extensional (B → B) ℓs ⦄
  → (f : A → B)
  → (g : B → A)
  → {@(tactic trivial-worker r (g ∘ f) id) p : Pathᵉ r (g ∘ f) id}
  → {@(tactic trivial-worker s (f ∘ g) id) q : Pathᵉ s (f ∘ g) id}
  → Iso A B
trivial-iso! ⦃ r ⦄ ⦃ s ⦄ f g {p = p} {q = q} =
  f , iso g (happly (s .idsᵉ .to-path q)) (happly (r .idsᵉ .to-path p))

abstract
  unext : ∀ {ℓ ℓr} {A : Type ℓ} ⦃ e : Extensional A ℓr ⦄ {x y : A} → x ≡ y → e .Pathᵉ x y
  unext ⦃ e ⦄ {x = x} p = transport (λ i → e .Pathᵉ x (p i)) (e .reflᵉ x)

macro
  reext!
    : ∀ {ℓ ℓr} {A : Type ℓ} ⦃ ea : Extensional A ℓr ⦄ {x y : A}
    → x ≡ y → Term → TC ⊤
  reext! p goal = do
    `p ← quoteTC p
    unify goal (def (quote ext) [ argN (def (quote unext) [ argN `p ]) ])

Pathᵉ-is-hlevel
  : ∀ {ℓ ℓr} {A : Type ℓ} n (sa : Extensional A ℓr)
  → is-hlevel A (suc n)
  → ∀ {x y}
  → is-hlevel (Pathᵉ sa x y) n
Pathᵉ-is-hlevel n sa hl =
  Equiv→is-hlevel _ (identity-system-gives-path (sa .idsᵉ)) (Path-is-hlevel' _ hl _ _)

-- Constructors for Extensional instances in terms of embeddings. The
-- extra coherence is required to make sure that we still have an
-- identity system by the end.
-- If the type you're reducing to is a set, use injection→extensional! instead.

embedding→extensional
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
  → (f : A ↪ B)
  → Extensional B ℓr
  → Extensional A ℓr
embedding→extensional f ext .Pathᵉ x y = Pathᵉ ext (f .fst x) (f .fst y)
embedding→extensional f ext .reflᵉ x = reflᵉ ext (f .fst x)
embedding→extensional f ext .idsᵉ =
  pullback-identity-system (ext .idsᵉ) f

iso→extensional
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
  → Iso A B
  → Extensional B ℓr
  → Extensional A ℓr
iso→extensional f ext =
  embedding→extensional (Iso→Embedding f) ext

injection→extensional
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
  → is-set B
  → {f : A → B}
  → (∀ {x y} → f x ≡ f y → x ≡ y)
  → Extensional B ℓr
  → Extensional A ℓr
injection→extensional b-set {f} inj ext .Pathᵉ x y = ext .Pathᵉ (f x) (f y)
injection→extensional b-set {f} inj ext .reflᵉ x = ext .reflᵉ (f x)
injection→extensional b-set {f} inj ext .idsᵉ .to-path x = inj (ext .idsᵉ .to-path x)
injection→extensional b-set {f} inj ext .idsᵉ .to-path-over p =
  is-prop→pathp (λ i → Pathᵉ-is-hlevel 1 ext b-set) _ _

injection→extensional!
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
  → ⦃ _ : H-Level B 2 ⦄
  → {f : A → B}
  → (∀ {x y} → f x ≡ f y → x ≡ y)
  → Extensional B ℓr
  → Extensional A ℓr
injection→extensional! = injection→extensional (hlevel 2)

Σ-prop-extensional
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : A → Type ℓ'}
  → (∀ x → is-prop (B x))
  → Extensional A ℓr
  → Extensional (Σ A B) ℓr
Σ-prop-extensional bprop = embedding→extensional (fst , Subset-proj-embedding bprop)

instance
  Extensional-Σ-trunc
    : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : A → Type ℓ'}
    → ⦃ ea : Extensional A ℓr ⦄ → Extensional (Σ A λ x → ∥ B x ∥) ℓr
  Extensional-Σ-trunc ⦃ ea ⦄ = Σ-prop-extensional (λ x → hlevel 1) ea

  Extensional-equiv
    : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
    → ⦃ ea : Extensional (A → B) ℓr ⦄ → Extensional (A ≃ B) ℓr
  Extensional-equiv ⦃ ea ⦄ = Σ-prop-extensional (λ x → hlevel 1) ea

  Extensional-emb
    : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
    → ⦃ ea : Extensional (A → B) ℓr ⦄ → Extensional (A ↪ B) ℓr
  Extensional-emb ⦃ ea ⦄ = Σ-prop-extensional (λ x → hlevel 1) ea

  Extensional-tr-map
    : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
    → ⦃ bset : H-Level B 2 ⦄
    → ⦃ ea : Extensional (A → B) ℓr ⦄
    → Extensional (∥ A ∥ → B) ℓr
  Extensional-tr-map ⦃ ea = ea ⦄ =
    injection→extensional! {f = λ f → f ∘ inc}
      (λ p → funext $ ∥-∥-elim (λ _ → hlevel 1) (happly p)) ea

private module test where
  _ : {f g : A → B} → ((x : A) → f x ≡ g x) → f ≡ g
  _ = ext

  _ : {f g : A × B → C} → ((a : A) (b : B) → f (a , b) ≡ g (a , b)) → f ≡ g
  _ = ext

  _ : {f g : Σ A P → B} → ((a : A) (b : P a) → f (a , b) ≡ g (a , b)) → f ≡ g
  _ = ext

  open Lift

  _ : {f g : Σ (Lift ℓ' A) (λ x → Lift ℓ' (P (x .lower))) → Lift ℓ'' B}
    → ((a : A) (b : P a) → f (lift a , lift b) .lower ≡ g (lift a , lift b) .lower)
    → f ≡ g
  _ = ext

  _ : {f g : A → B ≃ C} → ((a : A) (b : B) → f a .fst b ≡ g a .fst b) → f ≡ g
  _ = ext
