open import 1Lab.Reflection.Subst using (applyTC ; raiseTC)
open import 1Lab.Path.IdentitySystem
open import 1Lab.Reflection.HLevel
open import 1Lab.HLevel.Retracts
open import 1Lab.Reflection
open import 1Lab.Type.Sigma
open import 1Lab.Type.Pi
open import 1Lab.HLevel
open import 1Lab.Equiv
open import 1Lab.Path
open import 1Lab.Type

open import Data.Nat.Base

module 1Lab.Extensionality where

record Extensional {ℓ} (A : Type ℓ) (ℓ-rel : Level) : Typeω where
  no-eta-equality
  field
    Pathᵉ : A → A → Type ℓ-rel
    reflᵉ : ∀ x → Pathᵉ x x
    idsᵉ : is-identity-system Pathᵉ reflᵉ

record Extensionality {ℓ} (A : Type ℓ) : Type where
  field lemma : Name

{-
The 'Extensional' and 'Extensionality' classes both function to equip a
type with a preferred choice of identity system. The idea is that
'Extensional' actually carries the data, and 'Extensionality' is a
pointer to the 'Extensional' instance.

We use tactic arguments to implement default instances: since Agda will
happily call tactic arguments inside getInstances, if we had Extensional
instances with Extensional superclasses, this would lead to a
super-exponential amount of work being done: *every Extensional
instance*, including the "instance context", would have its full
instantiation computed before being added to the list of possible
instances.

Instead the macro looks for 'Extensionality' instances, which *have no
'Extensionality' class constraints*, and unfolds those to get the
"actual" 'Extensional' instance.
-}

open Extensional using (Pathᵉ ; reflᵉ ; idsᵉ) public

-- Default instance, uses regular paths for the relation.
Extensional-default : ∀ {ℓ} {A : Type ℓ} → Extensional A ℓ
Extensional-default .Pathᵉ   = _≡_
Extensional-default .reflᵉ _ = refl
Extensional-default .idsᵉ    = Path-identity-system

-- Find a value of 'Extensional' by looking at instances of 'Extensionality'.
find-extensionality : Term → TC Term
find-extensionality tm = do
  -- We have to block on the full type being available to prevent a
  -- situation where the default instance (or an incorrect instance!) is
  -- picked because the type is meta-headed.
  tm ← reduce =<< wait-for-type tm
  let search = def (quote Extensionality) [ argN tm ]
  debugPrint "tactic.extensionality" 10 ("find-extensionality goal:\n  " ∷ termErr search ∷ [])

  runSpeculative do
    (mv , _) ← new-meta' search
    soln ← getInstances mv >>= λ where
      -- In a throw-away TC context, look for solutions to 'Extensionality'
      -- tm, and choose the first instance if any are available.
      (x ∷ _) → do
        it ← unquoteTC {A = Name} =<< normalise (def (quote Extensionality.lemma) (argN x ∷ []))
        debugPrint "tactic.extensionality" 10 (" ⇒ found lemma " ∷ termErr (def it []) ∷ [])
        pure (def it [])

      -- If nothing more specific is available, use paths.
      [] → do
        debugPrint "tactic.extensionality" 10 " ⇒ using default"
        pure (def (quote Extensional-default) [])
    pure (soln , false)

-- Entry point for getting hold of an 'Extensional' instance:
extensional : ∀ {ℓ} (A : Type ℓ) → Term → TC ⊤
extensional A goal = do
  `A ← quoteTC A
  checkType goal (def (quote Extensional) [ argN `A , argN unknown ])
  unify goal =<< find-extensionality `A

{-
Unlike extensional, which is parametrised by a type, extensionalᶠ
can be parametrised by a function (of arbitrary arity) into types,
even though its type doesn't properly reflect this. It will discharge
goals like

   ∀ x → Extensional (P x) ℓ

by getting rid of every Π in the type and looking in an extended
context. It's also possible to use ⦃ _ : ∀ {x} → Extensional (P x) ℓ ⦄,
since Agda supports commuting those by itself. However, using
extensionalᶠ blocks more aggressively, which is required for touchy
instances like Extensional-natural-transformation.
-}
extensionalᶠ
  : ∀ {ℓ} {A : Type ℓ} → A → Term → TC ⊤
extensionalᶠ {A = A} fun goal = ⦇ wrap (quoteTC A) (quoteTC fun) ⦈ >>= id where
  work : Term → Term → TC Term
  work (pi dom@(arg ai _) (abs nm cod)) tm = do
    prf ← extendContext nm dom do
      tm ← raiseTC 1 tm
      tm ← applyTC tm (arg ai (var 0 []))
      work cod tm
    pure (lam (ai .ArgInfo.arg-vis) (abs nm prf))
  work _ tm = find-extensionality tm

  wrap : Term → Term → TC ⊤
  wrap t fn = do
    t ← wait-for-type t
    debugPrint "tactic.extensionality" 10 ("extensionalᶠ goal:\n  " ∷ termErr fn ∷ "\nof type\n  " ∷ termErr t ∷ [])
    prf ← work t fn
    unify goal prf

instance
  extensional-extensionality
    : ∀ {ℓ ℓs} {A : Type ℓ} {@(tactic extensional A) sa : Extensional A ℓs}
    → Extensional A ℓs
  extensional-extensionality {sa = sa} = sa

{-
Canonical example of extending the Extensionality tactic:

* The actual 'Extensional' value can have ⦃ Extensional A ℓ ⦄ in its
  "instance context". These are filled in by the bridge instance above.

* The 'Extensionality' loop-breaker only mentions the things that are
  actually required to compute the "head" type.
-}
Extensional-Lift
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ}
  → ⦃ sa : Extensional A ℓr ⦄
  → Extensional (Lift ℓ' A) ℓr
Extensional-Lift ⦃ sa ⦄ .Pathᵉ (lift x) (lift y) = sa .Pathᵉ x y
Extensional-Lift ⦃ sa ⦄ .reflᵉ (lift x) = sa .reflᵉ x
Extensional-Lift ⦃ sa ⦄ .idsᵉ .to-path p = ap lift (sa .idsᵉ .to-path p)
Extensional-Lift ⦃ sa ⦄ .idsᵉ .to-path-over p = sa .idsᵉ .to-path-over p

instance
  extensionality-lift : ∀ {ℓ ℓ'} {A : Type ℓ} → Extensionality (Lift ℓ' A)
  extensionality-lift = record { lemma = quote Extensional-Lift }

Extensional-Π
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : A → Type ℓ'}
  → ⦃ sb : ∀ {x} → Extensional (B x) ℓr ⦄
  → Extensional ((x : A) → B x) (ℓ ⊔ ℓr)
Extensional-Π ⦃ sb ⦄ .Pathᵉ f g = ∀ x → Pathᵉ sb (f x) (g x)
Extensional-Π ⦃ sb ⦄ .reflᵉ f x = reflᵉ sb (f x)
Extensional-Π ⦃ sb ⦄ .idsᵉ .to-path h = funext λ i → sb .idsᵉ .to-path (h i)
Extensional-Π ⦃ sb ⦄ .idsᵉ .to-path-over h = funextP λ i → sb .idsᵉ .to-path-over (h i)

Extensional-→
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
  → ⦃ sb : Extensional B ℓr ⦄
  → Extensional (A → B) (ℓ ⊔ ℓr)
Extensional-→ ⦃ sb ⦄ = Extensional-Π ⦃ λ {_} → sb ⦄

Extensional-×
  : ∀ {ℓ ℓ' ℓr ℓs} {A : Type ℓ} {B : Type ℓ'}
  → ⦃ sa : Extensional A ℓr ⦄
  → ⦃ sb : Extensional B ℓs ⦄
  → Extensional (A × B) (ℓr ⊔ ℓs)
Extensional-× ⦃ sa ⦄ ⦃ sb ⦄ .Pathᵉ (x , y) (x' , y') = Pathᵉ sa x x' × Pathᵉ sb y y'
Extensional-× ⦃ sa ⦄ ⦃ sb ⦄ .reflᵉ (x , y) = reflᵉ sa x , reflᵉ sb y
Extensional-× ⦃ sa ⦄ ⦃ sb ⦄ .idsᵉ .to-path (p , q) = ap₂ _,_
  (sa .idsᵉ .to-path p)
  (sb .idsᵉ .to-path q)
Extensional-× ⦃ sa ⦄ ⦃ sb ⦄ .idsᵉ .to-path-over (p , q) = Σ-pathp-dep
  (sa .idsᵉ .to-path-over p)
  (sb .idsᵉ .to-path-over q)

instance
  extensionality-fun
    : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'}
    → Extensionality (A → B)
  extensionality-fun = record { lemma = quote Extensional-→ }

  extensionality-Π
    : ∀ {ℓ ℓ'} {A : Type ℓ} {B : A → Type ℓ'}
    → Extensionality ((x : A) → B x)
  extensionality-Π = record { lemma = quote Extensional-Π }

  extensionality-×
    : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'}
    → Extensionality (A × B)
  extensionality-× = record { lemma = quote Extensional-× }

{-
Actual user-facing entry point for the tactic: using the 'extensional'
tactic (through the blanket instance) we can find an identity system for
the type A, and turn a proof in the computed relation to an identity.
-}
ext
  : ∀ {ℓ ℓr} {A : Type ℓ} ⦃ r : Extensional A ℓr ⦄ {x y : A}
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
    : ∀ {ℓ ℓr} {A : Type ℓ}
    → (r : Extensional A ℓr)
    → (x y : A)
    → Term
    → TC ⊤
  trivial-worker r x y goal = try where
    error : ∀ {ℓ} {A : Type ℓ} → TC A

    -- We already have our r : Extensional A ℓr, and this is a macro, so
    -- we can just check that r .reflᵉ x : R x y. If that's the case
    -- then we can use that as the argument, otherwise we can give a
    -- slightly better error message than what Agda would yell.
    try : TC ⊤
    try = do
      `r ← wait-for-type =<< quoteωTC r
      ty ← quoteTC (Pathᵉ r x y)
      `x ← quoteTC x
      `refl ← checkType (def (quote reflᵉ) [ argN `r , argN `x ]) ty
        <|> error
      unify goal `refl

    error = do
      `x ← quoteTC x
      `y ← quoteTC y
      typeError
        [ "trivial! failed: the values\n  "
        , termErr `x
        , "\nand\n  "
        , termErr `y
        , "\nare not extensionally equal by refl."
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

Pathᵉ-is-hlevel
  : ∀ {ℓ ℓr} {A : Type ℓ} n (sa : Extensional A ℓr)
  → is-hlevel A (suc n)
  → ∀ {x y}
  → is-hlevel (Pathᵉ sa x y) n
Pathᵉ-is-hlevel n sa hl =
  is-hlevel≃ _ (identity-system-gives-path (sa .idsᵉ)) (Path-is-hlevel' _ hl _ _)

{-
Convenient wrapper to define an Extensional instance if we're given
an injection 'f : A → B' into a set. It would be possible to take an
embedding into an arbitrary type, but that hasn't come up yet. The
relation this equips A with is "equal under f".
-}
injection→extensional!
  : ∀ {ℓ ℓ' ℓr} {A : Type ℓ} {B : Type ℓ'}
  → {@(tactic hlevel-tactic-worker) sb : is-set B}
  → {f : A → B}
  → (∀ {x y} → f x ≡ f y → x ≡ y)
  → Extensional B ℓr
  → Extensional A ℓr
injection→extensional! {sb = b-set} {f} inj ext .Pathᵉ x y = Pathᵉ ext (f x) (f y)
injection→extensional! {sb = b-set} {f} inj ext .reflᵉ x = reflᵉ ext (f x)
injection→extensional! {sb = b-set} {f} inj ext .idsᵉ =
  set-identity-system
    (λ x y → Pathᵉ-is-hlevel 1 ext b-set)
    (λ p → inj (ext .idsᵉ .to-path p))