<!--
```agda
open import 1Lab.Path.Groupoid
open import 1Lab.Path
open import 1Lab.Type
```
-->

```agda
module 1Lab.Path.Reasoning where
```

# Reasoning combinators for path spaces

<!--
```agda
private variable
  ℓ : Level
  A : Type ℓ
  x y : A
  p p' q q' r r' s s' t u v : x ≡ y

∙-filler''
  : ∀ {ℓ} {A : Type ℓ} {x y z : A} (p : x ≡ y) (q : y ≡ z)
  → Square refl (sym p) q (p ∙ q)
∙-filler'' {x = x} {y} {z} p q i j =
  hcomp (∂ i ∨ ~ j) λ where
    k (i = i0) → p (~ j)
    k (i = i1) → q (j ∧ k)
    k (j = i0) → y
    k (k = i0) → p (i ∨ ~ j)

pasteP
  : ∀ {ℓ} {A : Type ℓ} {w w' x x' y y' z z' : A}
    {p p' q q' r r' s s'}
    {α β γ δ}
  → Square α p p' β
  → Square α q q' γ
  → Square β r r' δ
  → Square γ s s' δ
  → Square {a00 = w}  {x}  {y}  {z}  p  q  r  s
  → Square {a00 = w'} {x'} {y'} {z'} p' q' r' s'
pasteP top left right bottom square i j = hcomp (∂ i ∨ ∂ j) λ where
  k (i = i0) → left k j
  k (i = i1) → right k j
  k (j = i0) → top k i
  k (j = i1) → bottom k i
  k (k = i0) → square i j

paste
  : p ≡ p' → q ≡ q' → r ≡ r' → s ≡ s'
  → Square p q r s
  → Square p' q' r' s'
paste p q r s = pasteP p q r s
```
-->

## Identity paths

```agda
∙-id-comm : p ∙ refl ≡ refl ∙ p
∙-id-comm {p = p} i j =
  hcomp (∂ i ∨ ∂ j) λ where
    k (i = i0) → ∙-filler p refl k j
    k (i = i1) → ∙-filler'' refl p j k
    k (j = i0) → p i0
    k (j = i1) → p (~ i ∨ k)
    k (k = i0) → (p (~ i ∧ j))

module _ (p≡refl : p ≡ refl) where opaque
  ∙-eliml : p ∙ q ≡ q
  ∙-eliml {q = q} = sym $ paste (ap sym p≡refl) refl refl refl (∙-filler' p q)

  ∙-elimr : q ∙ p ≡ q
  ∙-elimr {q = q} = sym $ paste refl refl refl p≡refl (∙-filler q p)

  ∙-introl : q ≡ p ∙ q
  ∙-introl = sym ∙-eliml

  ∙-intror : q ≡ q ∙ p
  ∙-intror = sym ∙-elimr
```

## Reassocations

We often find ourselves in situations where we have an equality
involving the composition of 2 morphisms, but the association
is a bit off. These combinators aim to address that situation.

```agda
module _ (pq≡s : p ∙ q ≡ s) where
  ∙-pulll : p ∙ q ∙ r ≡ s ∙ r
  ∙-pulll {r = r} = ∙-assoc p q r ∙ ap (_∙ r) pq≡s

  double-left : p ·· q ·· r ≡ s ∙ r
  double-left {r = r} = double-composite p q r ∙ ∙-pulll

  ∙-pullr : (r ∙ p) ∙ q ≡ r ∙ s
  ∙-pullr {r = r} = sym (∙-assoc r p q) ∙ ap (r ∙_) pq≡s

  ∙-swapl : q ≡ sym p ∙ s
  ∙-swapl = ∙-introl (∙-invl p) ∙ ∙-pullr

  ∙-swapr : p ≡ s ∙ sym q
  ∙-swapr = ∙-intror (∙-invr q) ∙ ∙-pulll

module _ (s≡pq : s ≡ p ∙ q) where
  ∙-pushl : s ∙ r ≡ p ∙ q ∙ r
  ∙-pushl = sym (∙-pulll (sym s≡pq))

  ∙-pushr : r ∙ s ≡ (r ∙ p) ∙ q
  ∙-pushr = sym (∙-pullr (sym s≡pq))

  ∙→square : Square refl p s q
  ∙→square = ∙-filler p q ▷ sym s≡pq

  ∙→square' : Square (sym p) q s refl
  ∙→square' = ∙-filler' p q ▷ sym s≡pq

  ∙→square'' : Square (sym p) refl s q
  ∙→square'' = transpose (∙-filler'' p q) ▷ sym s≡pq

module _ (pq≡rs : p ∙ q ≡ r ∙ s) where
  ∙-extendl : p ∙ (q ∙ t) ≡ r ∙ (s ∙ t)
  ∙-extendl {t = t} = ∙-assoc _ _ _ ·· ap (_∙ t) pq≡rs ·· sym (∙-assoc _ _ _)

  ∙-extendr : (t ∙ p) ∙ q ≡ (t ∙ r) ∙ s
  ∙-extendr {t = t} = sym (∙-assoc _ _ _) ·· ap (t ∙_) pq≡rs ·· ∙-assoc _ _ _

··-stack : (sym p' ·· (sym p ·· q ·· r) ·· r') ≡ (sym (p ∙ p') ·· q ·· (r ∙ r'))
··-stack = ··-unique' (··-filler _ _ _ ∙₂ ··-filler _ _ _)

··-chain : (sym p ·· q ·· r) ∙ (sym r ·· q' ·· s) ≡ sym p ·· (q ∙ q') ·· s
··-chain {p = p} {q = q} {r = r} {q' = q'} {s = s} = sym (∙-unique _ square) where
  square : Square refl (sym p ·· q ·· r) (sym p ·· (q ∙ q') ·· s) (sym r ·· q' ·· s)
  square i j = hcomp (~ j ∨ (j ∧ (i ∨ ~ i))) λ where
    k (k = i0) → ∙-filler q q' i j
    k (j = i0) → p k
    k (j = i1) (i = i0) → r k
    k (j = i1) (i = i1) → s k

··-∙-assoc : p ·· q ·· (r ∙ s) ≡ (p ·· q ·· r) ∙ s
··-∙-assoc {p = p} {q = q} {r = r} {s = s} = sym (··-unique' square) where
  square : Square (sym p) q ((p ·· q ·· r) ∙ s) (r ∙ s)
  square i j = hcomp (~ i ∨ ~ j ∨ (i ∧ j)) λ where
    k (k = i0) → ··-filler p q r i j
    k (i = i0) → q j
    k (j = i0) → p (~ i)
    k (i = i1) (j = i1) → s k

module _ {p q : x ≡ y} where
  ∧-square : p ≡ q → Square p q refl refl
  ∧-square sq i j = hcomp (∂ i ∨ ∂ j) λ where
    k (k = i0) → p j
    k (i = i0) → sq k j
    k (i = i1) → p (j ∨ k)
    k (j = i0) → p (i ∧ k)
    k (j = i1) → y

  flatten-∧-square : Square p q refl refl → p ≡ q
  flatten-∧-square sq i j = hcomp (∂ i ∨ ∂ j) λ where
    k (k = i0) → sq j i
    k (i = i0) → p j
    k (i = i1) → q (j ∨ ~ k)
    k (j = i1) → y
    k (j = i0) → q (i ∧ ~ k)

  ∨-square : p ≡ q → Square refl refl p q
  ∨-square sq i j = hcomp (∂ i ∨ ∂ j) λ where
    k (k = i0) → p (i ∧ j)
    k (i = i0) → x
    k (i = i1) → p j
    k (j = i0) → x
    k (j = i1) → sq k i

  flatten-∨-square : Square refl refl p q → p ≡ q
  flatten-∨-square sq i j = hcomp (∂ i ∨ ∂ j) λ where
    k (k = i0) → sq j i
    k (i = i0) → p (j ∧ k)
    k (i = i1) → q j
    k (j = i0) → x
    k (j = i1) → p (i ∨ k)
```

## Cancellation

These lemmas do 2 things at once: rearrange parenthesis, and also remove
things that are equal to `id`.

```agda
module _ (inv : p ∙ q ≡ refl) where abstract
  ∙-cancelsl : p ∙ (q ∙ r) ≡ r
  ∙-cancelsl = ∙-pulll inv ∙ ∙-idl _

  ∙-cancelsr : (r ∙ p) ∙ q ≡ r
  ∙-cancelsr = ∙-pullr inv ∙ ∙-idr _

  ∙-insertl : r ≡ p ∙ (q ∙ r)
  ∙-insertl = sym ∙-cancelsl

  ∙-insertr : r ≡ (r ∙ p) ∙ q
  ∙-insertr = sym ∙-cancelsr
```

## Notation

When doing equational reasoning, it's often somewhat clumsy to have to write
`ap (f ∙_) p` when proving that `f ∙ g ≡ f ∙ h`. To fix this, we steal
some cute mixfix notation from `agda-categories` which allows us to write
`≡⟨ refl⟩∙⟨ p ⟩` instead, which is much more aesthetically pleasing!

```agda
_⟩∙⟨_ : p ≡ q → r ≡ s → p ∙ r ≡ q ∙ s
_⟩∙⟨_ p q i = p i ∙ q i

refl⟩∙⟨_ : p ≡ q → r ∙ p ≡ r ∙ q
refl⟩∙⟨_ {r = r} p = ap (r ∙_) p

_⟩∙⟨refl : p ≡ q → p ∙ r ≡ q ∙ r
_⟩∙⟨refl {r = r} p = ap (_∙ r) p
```
