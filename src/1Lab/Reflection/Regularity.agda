{-# OPTIONS --allow-unsolved-metas -vtactic:10 #-}
open import 1Lab.Reflection.HLevel
open import 1Lab.Reflection.Subst
open import 1Lab.Reflection
open import 1Lab.Path
open import 1Lab.Type

module 1Lab.Reflection.Regularity where

{-
A tactic for reducing "transport refl x" in other-wise normal terms. The
implementation is actually surprisingly simple: A term of the form (e.g.)

    transp (λ _ → A) i0 (f (transp (λ _ → B) i0 x))

is already a blueprint for how to normalise it. We simply have to turn it into

    λ i → transp (λ _ → A) i (f (transp (λ _ → B) i x))

so that the constant transports reduce away when (i = i1). Abstracting over i,
this gives a path from the initial term to its "regular normal form", which
may be the worst name ever.

More generally, we replace terms of the form `transp Al φ x` with `transp Al (φ ∨ i) x`
recursively (inside-out), on the condition that Al is constant when i = i1.
-}

private
  -- We have a double composition operator that doesn't use the
  -- fancy hcomp syntax in its definition. This has better type
  -- inference for one of the macros since it guarantees that the base
  -- (q i) is independent of j without any reduction.
  double-comp
    : ∀ {ℓ} {A : Type ℓ} {w z : A} (x y : A)
    → w ≡ x → x ≡ y → y ≡ z
    → w ≡ z
  double-comp x y p q r i = primHComp
    (λ { j (i = i0) → p (~ j) ; j (i = i1) → r j }) (q i)

  -- The regularity tactic can operate in two modes: `precise` will work
  -- with the type-checker to identify which `transp`s are along refl,
  -- and which should be preserved. The `fast` mode says YOLO and
  -- assumes that **every** application of `transp` is one that would
  -- reduce by regularity. Needless to say, only use `fast` when you're
  -- sure that's the case (e.g. the fibres of a displayed category over
  -- Sets)
  data Regularity-precision : Type where
    precise fast : Regularity-precision
  -- As the name implies, `precise` is `precise`, while `fast` is
  -- `fast`. The reason is that `fast` will avoid traversing many of the
  -- terms involved in a `transp`: It doesn't care about the level, it
  -- doesn't care about the line, and it doesn't care about the
  -- cofibration.

  -- The core of the tactic is this triad of mutually recursive
  -- functions. In all three of them, the `Nat` argument indicates how
  -- many binders we've gone under: it is the dimension variable we
  -- abstracted over at the start.
  refl-transport : Nat → Term → TC Term
  -- ^ Determines whether an application of `transp` is a case of
  -- regularity, and if so, replaces it by `regular`. Precondition: its
  -- subterms must already be reduced.
  go  : Regularity-precision → Nat → Term → TC Term
  -- ^ Reduces all the subterms, and finds applications of `transp` to
  -- hand off to `refl-transport`.
  go* : Regularity-precision → Nat → List (Arg Term) → TC (List (Arg Term))
  -- ^ Isn't the termination checker just lovely?

  refl-transport n tm@(def (quote transp) (ℓ h∷ Al v∷ φ v∷ x v∷ [])) =
    -- This match might make you wonder: Can't Al be a line of
    -- functions, so that the transport will have more arguments? No:
    -- The term is in normal form.
    (do
      debugPrint "tactic.regularity" 10 $ "Checking regularity of " ∷ termErr tm ∷ []
      let φ' = def (quote _∨_) (φ v∷ var n [] v∷ [])
      let tm' = def (quote transp) (ℓ h∷ Al v∷ φ' v∷ x v∷ [])
      -- We simply ask Agda to check that the newly constructed term `transp Al (φ ∨ i) x`
      -- is correct, i.e. that Al is constant on (i = i1).
      -- If it isn't, we backtrack and leave the term unchanged.
      -- Note that if Al itself contains constant transports, we have already processed those,
      -- so they reduce away when (i = i1).
      check-type tm' unknown -- infer-type doesn't trigger the constancy check https://github.com/agda/agda/issues/6585
      pure tm') <|>
    (do
      debugPrint "tactic.regularity" 10 $ "NOT a (transport refl): " ∷ termErr tm ∷ []
      pure tm)
  refl-transport _ tm = pure tm

  -- Boring term traversal.
  go pre n (var x args) = var x <$> go* pre n args
  go pre n (con c args) = con c <$> go* pre n args
  go fast n (def (quote transp) (ℓ h∷ Al v∷ φ v∷ x v∷ [])) = do
    x ← go fast n x
    let φ' = def (quote _∨_) (φ v∷ var n [] v∷ [])
    pure $ def (quote transp) (ℓ h∷ Al v∷ φ' v∷ x v∷ [])
  go pre n (def f args) = do
    as ← go* pre n args
    refl-transport n (def f as)
  go pre k t@(lam v (abs nm b)) = lam v ∘ abs nm <$> under-abs t (go pre (suc k) b)
  go pre n (pat-lam cs args) = typeError $ "regularity: Can not deal with pattern lambdas"
  go pre n t@(pi (arg i a) (abs nm b)) = do
    a ← go pre n a
    b ← under-abs t (go pre (suc n) b)
    pure (pi (arg i a) (abs nm b))
  go pre n (agda-sort s) = pure (agda-sort s)
  go pre n (lit l) = pure (lit l)
  go pre n (meta x args) = meta x <$> go* pre n args
  go pre n unknown = pure unknown

  go* pre n [] = pure []
  go* pre n (arg i a ∷ as) = do
    a ← go pre n a
    (arg i a ∷_) <$> go* pre n as

  -- To turn a term into a regularity path, given a level of precision,
  -- all we have to do is raise the term by one, do the procedure above,
  -- then wrap it in a lambda. Nice!
  to-regularity-path : Regularity-precision → Term → TC Term
  to-regularity-path pre tm = do
    let tm = raise 1 tm
    -- Since we'll be comparing terms, Agda really wants them to be
    -- well-scoped. Since we shifted eeeverything up by one, we have to
    -- grow the context, too.
    tm ← run-speculative $ extend-context "i" (argN (quoteTerm I)) do
      tm ← go pre 0 tm
      pure (tm , false)
    pure $ vlam "i" tm

  -- Extend a path x ≡ y to a path x' ≡ y', where x' --> x and y' --> y
  -- under the given regularity precision. Shorthand for composing
  --    regularity! ∙ p ∙ sym regularity!.
  regular!-worker :
    ∀ {ℓ} {A : Type ℓ} {x y : A}
    → Regularity-precision
    → x ≡ y
    → Term
    → TC ⊤
  regular!-worker {x = x} {y} pre p goal = do
    gt ← infer-type goal
    `x ← quoteTC x
    `y ← quoteTC y
    `p ← quoteTC p
    just (_ , l , r) ← unapply-path =<< wait-for-type =<< infer-type goal
      where _ → typeError [ "goal type is not path type: " , termErr goal ]
    l ← normalise =<< wait-for-type l
    r ← normalise =<< wait-for-type r
    reg ← to-regularity-path pre l
    reg' ← to-regularity-path pre r
    unify-loudly goal $ def (quote double-comp) $
         `x v∷ `y v∷ reg
      v∷ `p
      v∷ def (quote sym) (reg' v∷ [])
      v∷ []

module Regularity where
  open Regularity-precision public
  -- The reflection interface: Regularity.reduce! will, well, reduce a
  -- term. There's a lot of blocking involved in making this work.
  macro
    reduce! : Term → TC ⊤
    reduce! goal = withNormalisation true do
      just (_ , l , r) ← unapply-path =<< wait-for-type =<< infer-type goal
        where _ → typeError []
      reg ← to-regularity-path precise l
      unify-loudly goal reg

    -- We then have wrappers that reduce on one side, and expand on the
    -- other, depending on how precise you want the reduction to be.
    precise! : ∀ {ℓ} {A : Type ℓ} {x y : A} → x ≡ y → Term → TC ⊤
    fast! : ∀ {ℓ} {A : Type ℓ} {x y : A} → x ≡ y → Term → TC ⊤

    precise! p goal = regular!-worker precise p goal
    fast! p goal = regular!-worker fast p goal

    -- For debugging purposes, this macro will take a term and output
    -- its (transport refl)-normal form, according to the given level of
    -- precision.
    reduct : Regularity-precision → Term → Term → TC ⊤
    reduct pres tm _ = do
      orig ← wait-for-type =<< normalise tm
      tm ← to-regularity-path pres orig
      red ← normalise (apply-tm tm (argN (con (quote i1) [])))
      `pres ← quoteTC pres
      typeError $
        "The term\n\n  " ∷ termErr orig ∷ "\n\nreduces modulo " ∷ termErr `pres ∷ " regularity to\n\n  "
        ∷ termErr red
        ∷ "\n"

-- Test cases.
module
  _ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (f g : A → B) (x : A)
    (a-loop : (i : I) → Type ℓ [ (i ∨ ~ i) ↦ (λ ._ → A) ])
  where private

  p : (h : ∀ x → f x ≡ g x)
    → transport (λ i → A → B)
        (λ x → transport refl (transport refl (f (transport refl x)))) ≡ g
  p h = Regularity.reduce! ∙ funext h

  q : transport refl (transp (λ i → outS (a-loop i)) i0 x) ≡ transp (λ i → outS (a-loop i)) i0 x
  q = Regularity.reduce!

  -- Imprecise/fast reduction: According to it, the normal form of the
  -- transport below is refl. That's.. not the case, at least we don't
  -- know so. Precise regularity handles it, though.
  q' : ⊤
  q' = {! Regularity.reduct Regularity.fast (transp (λ i → outS (a-loop i)) i0 x) !}

  r : (h : ∀ x → f x ≡ g x) → transport refl (f x) ≡ transport refl (g x)
  r h = Regularity.precise! (h x)

  s : (h : ∀ x → f x ≡ g x) → transport refl (g x) ≡ transport refl (f x)
  s h = sym $ Regularity.fast! (h x)
