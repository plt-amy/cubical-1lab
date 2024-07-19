<!--
```agda
open import 1Lab.Prelude hiding (_∘_ ; id ; _↪_ ; _↠_)

open import Cat.Solver
open import Cat.Base
```
-->

```agda
module Cat.Morphism {o h} (C : Precategory o h) where
```

<!--
```agda
open Precategory C public
private variable
  a b c d : Ob
  f : Hom a b
```
-->

# Morphisms

This module defines the three most important classes of morphisms that
can be found in a category: **monomorphisms**, **epimorphisms**, and
**isomorphisms**.

## Monos {defines="monomorphism monic"}

A morphism is said to be **monic** when it is left-cancellable. A
**monomorphism** from $A$ to $B$, written $A \mono B$, is a monic
morphism.

```agda
is-monic : Hom a b → Type _
is-monic {a = a} f = ∀ {c} → (g h : Hom c a) → f ∘ g ≡ f ∘ h → g ≡ h

is-monic-is-prop : ∀ {a b} (f : Hom a b) → is-prop (is-monic f)
is-monic-is-prop f x y i {c} g h p = Hom-set _ _ _ _ (x g h p) (y g h p) i

record _↪_ (a b : Ob) : Type (o ⊔ h) where
  field
    mor   : Hom a b
    monic : is-monic mor

open _↪_ public
```

<!--
```agda
Factors : ∀ {A B C} (f : Hom A C) (g : Hom B C) → Type _
Factors f g = Σ[ h ∈ Hom _ _ ] (f ≡ g ∘ h)
```
-->

Conversely, a morphism is said to be **epic** when it is
right-cancellable.  An **epimorphism** from $A$ to $B$, written $A \epi
B$, is an epic morphism.

## Epis {defines="epimorphism epic"}

```agda
is-epic : Hom a b → Type _
is-epic {b = b} f = ∀ {c} → (g h : Hom b c) → g ∘ f ≡ h ∘ f → g ≡ h

is-epic-is-prop : ∀ {a b} (f : Hom a b) → is-prop (is-epic f)
is-epic-is-prop f x y i {c} g h p = Hom-set _ _ _ _ (x g h p) (y g h p) i

record _↠_ (a b : Ob) : Type (o ⊔ h) where
  field
    mor  : Hom a b
    epic : is-epic mor

open _↠_ public
```

The identity morphism is monic and epic.

```agda
id-monic : ∀ {a} → is-monic (id {a})
id-monic g h p = sym (idl _) ·· p ·· idl _

id-epic : ∀ {a} → is-epic (id {a})
id-epic g h p = sym (idr _) ·· p ·· idr _
```

Both monos and epis are closed under composition.

```agda
monic-∘
  : ∀ {a b c} {f : Hom b c} {g : Hom a b}
  → is-monic f
  → is-monic g
  → is-monic (f ∘ g)
monic-∘ fm gm a b α = gm _ _ (fm _ _ (assoc _ _ _ ·· α ·· sym (assoc _ _ _)))

epic-∘
  : ∀ {a b c} {f : Hom b c} {g : Hom a b}
  → is-epic f
  → is-epic g
  → is-epic (f ∘ g)
epic-∘ fe ge a b α = fe _ _ (ge _ _ (sym (assoc _ _ _) ·· α ·· (assoc _ _ _)))

_∘Mono_ : ∀ {a b c} → b ↪ c → a ↪ b → a ↪ c
(f ∘Mono g) .mor = f .mor ∘ g .mor
(f ∘Mono g) .monic = monic-∘ (f .monic) (g .monic)

_∘Epi_ : ∀ {a b c} → b ↠ c → a ↠ b → a ↠ c
(f ∘Epi g) .mor = f .mor ∘ g .mor
(f ∘Epi g) .epic = epic-∘ (f .epic) (g .epic)
```

If $f \circ g$ is monic, then $g$ must also be monic.

```agda
monic-cancell
  : ∀ {a b c} {f : Hom b c} {g : Hom a b}
  → is-monic (f ∘ g)
  → is-monic g
monic-cancell {f = f} fg-monic h h' p = fg-monic h h' $
  sym (assoc _ _ _) ·· ap (f ∘_) p ·· assoc _ _ _
```

Dually, if $f \circ g$ is epic, then $f$ must also be epic.

```agda
epic-cancelr
  : ∀ {a b c} {f : Hom b c} {g : Hom a b}
  → is-epic (f ∘ g)
  → is-epic f
epic-cancelr {g = g} fg-epic h h' p = fg-epic h h' $
  assoc _ _ _ ·· ap (_∘ g) p ·· sym (assoc _ _ _)
```

Postcomposition with a mono is an embedding.

```agda
monic-postcomp-embedding
  : ∀ {a b c} {f : Hom b c}
  → is-monic f
  → is-embedding {A = Hom a b} (f ∘_)
monic-postcomp-embedding monic =
  injective→is-embedding (Hom-set _ _) _ (monic _ _)
```

Likewise, precomposition with an epi is an embedding.

```agda
epic-precomp-embedding
  : ∀ {a b c} {f : Hom a b}
  → is-epic f
  → is-embedding {A = Hom b c} (_∘ f)
epic-precomp-embedding epic =
  injective→is-embedding (Hom-set _ _) _ (epic _ _)
```

## Sections

A morphism $s : B \to A$ is a section of another morphism $r : A \to B$
when $r \cdot s = id$. The intuition for this name is that $s$ picks
out a cross-section of $a$ from $b$. For instance, $r$ could map
animals to their species; a section of this map would have to pick out
a canonical representative of each species from the set of all animals.

```agda
_section-of_ : (s : Hom b a) (r : Hom a b) → Type _
s section-of r = r ∘ s ≡ id

section-of-is-prop
  : ∀ {s : Hom b a} {r : Hom a b}
  → is-prop (s section-of r)
section-of-is-prop = Hom-set _ _ _ _

record has-section (r : Hom a b) : Type h where
  constructor make-section
  field
    section : Hom b a
    is-section : section section-of r

open has-section public

id-has-section : ∀ {a} → has-section (id {a})
id-has-section .section = id
id-has-section .is-section = idl _

section-of-∘
  : ∀ {a b c} {f : Hom c b} {g : Hom b c} {h : Hom b a} {i : Hom a b}
  → f section-of g → h section-of i
  → (h ∘ f) section-of (g ∘ i)
section-of-∘ {f = f} {g = g} {h = h} {i = i} fg-sect hi-sect =
  (g ∘ i) ∘ h ∘ f ≡⟨ cat! C ⟩
  g ∘ (i ∘ h) ∘ f ≡⟨ ap (λ ϕ → g ∘ ϕ ∘ f) hi-sect ⟩
  g ∘ id ∘ f      ≡⟨ ap (g ∘_) (idl _) ⟩
  g ∘ f           ≡⟨ fg-sect ⟩
  id ∎

section-∘
  : ∀ {a b c} {f : Hom b c} {g : Hom a b}
  → has-section f → has-section g → has-section (f ∘ g)
section-∘ f-sect g-sect .section = g-sect .section ∘ f-sect .section
section-∘ {f = f} {g = g} f-sect g-sect .is-section =
  section-of-∘ (f-sect .is-section) (g-sect .is-section)
```

If $f$ has a section, then $f$ is epic.

```agda
has-section→epic
  : ∀ {a b} {f : Hom a b}
  → has-section f
  → is-epic f
has-section→epic {f = f} f-sect g h p =
  g                         ≡˘⟨ idr _ ⟩
  g ∘ id                    ≡˘⟨ ap (g ∘_) (f-sect .is-section) ⟩
  g ∘ f ∘ f-sect .section   ≡⟨ assoc _ _ _ ⟩
  (g ∘ f) ∘ f-sect .section ≡⟨ ap (_∘ f-sect .section) p ⟩
  (h ∘ f) ∘ f-sect .section ≡˘⟨ assoc _ _ _ ⟩
  h ∘ f ∘ f-sect .section   ≡⟨ ap (h ∘_) (f-sect .is-section) ⟩
  h ∘ id                    ≡⟨ idr _ ⟩
  h ∎
```

## Retracts

A morphism $r : A \to B$ is a retract of another morphism $s : B \to A$
when $r \cdot s = id$. Note that this is the same equation involved
in the definition of a section; retracts and sections always come in
pairs. If sections solve a sort of "curation problem" where we are
asked to pick out canonical representatives, then retracts solve a
sort of "classification problem".

```agda
_retract-of_ : (r : Hom a b) (s : Hom b a) → Type _
r retract-of s = r ∘ s ≡ id

retract-of-is-prop
  : ∀ {s : Hom b a} {r : Hom a b}
  → is-prop (r retract-of s)
retract-of-is-prop = Hom-set _ _ _ _

record has-retract (s : Hom b a) : Type h where
  constructor make-retract
  field
    retract : Hom a b
    is-retract : retract retract-of s

open has-retract public

id-has-retract : ∀ {a} → has-retract (id {a})
id-has-retract .retract = id
id-has-retract .is-retract = idl _

retract-of-∘
  : ∀ {a b c} {f : Hom c b} {g : Hom b c} {h : Hom b a} {i : Hom a b}
  → f retract-of g → h retract-of i
  → (h ∘ f) retract-of (g ∘ i)
retract-of-∘ fg-ret hi-ret = section-of-∘ hi-ret fg-ret

retract-∘
  : ∀ {a b c} {f : Hom b c} {g : Hom a b}
  → has-retract f → has-retract g
  → has-retract (f ∘ g)
retract-∘ f-ret g-ret .retract = g-ret .retract ∘ f-ret .retract
retract-∘ f-ret g-ret .is-retract =
  retract-of-∘ (f-ret .is-retract) (g-ret .is-retract)
```

If $f$ has a retract, then $f$ is monic.

```agda
has-retract→monic
  : ∀ {a b} {f : Hom a b}
  → has-retract f
  → is-monic f
has-retract→monic {f = f} f-ret g h p =
  g                        ≡˘⟨ idl _ ⟩
  id ∘ g                   ≡˘⟨ ap (_∘ g) (f-ret .is-retract) ⟩
  (f-ret .retract ∘ f) ∘ g ≡˘⟨ assoc _ _ _ ⟩
  f-ret .retract ∘ (f ∘ g) ≡⟨ ap (f-ret .retract ∘_) p ⟩
  f-ret .retract ∘ f ∘ h   ≡⟨ assoc _ _ _ ⟩
  (f-ret .retract ∘ f) ∘ h ≡⟨ ap (_∘ h) (f-ret .is-retract) ⟩
  id ∘ h                   ≡⟨ idl _ ⟩
  h                        ∎
```

A section that is also epic is a retract.

```agda
section-of+epic→retract-of
  : ∀ {a b} {s : Hom b a} {r : Hom a b}
  → s section-of r
  → is-epic s
  → s retract-of r
section-of+epic→retract-of {s = s} {r = r} sect epic =
  epic (s ∘ r) id $
    (s ∘ r) ∘ s ≡˘⟨ assoc s r s ⟩
    s ∘ (r ∘ s) ≡⟨ ap (s ∘_) sect ⟩
    s ∘ id      ≡⟨ idr _ ⟩
    s           ≡˘⟨ idl _ ⟩
    id ∘ s      ∎
```

Dually, a retract that is also monic is a section.

```agda
retract-of+monic→section-of
  : ∀ {a b} {s : Hom b a} {r : Hom a b}
  → r retract-of s
  → is-monic r
  → r section-of s
retract-of+monic→section-of {s = s} {r = r} ret monic =
  monic (s ∘ r) id $
    r ∘ s ∘ r   ≡⟨ assoc r s r ⟩
    (r ∘ s) ∘ r ≡⟨ ap (_∘ r) ret ⟩
    id ∘ r      ≡⟨ idl _ ⟩
    r           ≡˘⟨ idr _ ⟩
    r ∘ id      ∎
```

<!--
```agda
has-retract+epic→has-section
  : ∀ {a b} {f : Hom a b}
  → has-retract f
  → is-epic f
  → has-section f
has-retract+epic→has-section ret epic .section = ret .retract
has-retract+epic→has-section ret epic .is-section =
  section-of+epic→retract-of (ret .is-retract) epic

has-section+monic→has-retract
  : ∀ {a b} {f : Hom a b}
  → has-section f
  → is-monic f
  → has-retract f
has-section+monic→has-retract sect monic .retract = sect .section
has-section+monic→has-retract sect monic .is-retract =
  retract-of+monic→section-of (sect .is-section) monic
```
-->


## Isos {defines="isomorphism invertible"}

Maps $f : A \to B$ and $g : B \to A$ are **inverses** when we have $f
\circ g$ and $g \circ f$ both equal to the identity. A map $f : A \to B$
**is invertible** (or **is an isomorphism**) when there exists a $g$ for
which $f$ and $g$ are inverses. An **isomorphism** $A \cong B$ is a
choice of map $A \to B$, together with a specified inverse.

```agda
record Inverses (f : Hom a b) (g : Hom b a) : Type h where
  field
    invl : f ∘ g ≡ id
    invr : g ∘ f ≡ id

open Inverses

record is-invertible (f : Hom a b) : Type h where
  field
    inv : Hom b a
    inverses : Inverses f inv

  open Inverses inverses public

  op : is-invertible inv
  op .inv = f
  op .inverses .Inverses.invl = invr inverses
  op .inverses .Inverses.invr = invl inverses

record _≅_ (a b : Ob) : Type h where
  field
    to       : Hom a b
    from     : Hom b a
    inverses : Inverses to from

  open Inverses inverses public

open _≅_ public
```

A given map is invertible in at most one way: If you have $f : A \to B$
with two _inverses_ $g : B \to A$ and $h : B \to A$, then not only are
$g = h$ equal, but the witnesses of these equalities are irrelevant.

```agda
Inverses-are-prop : ∀ {f : Hom a b} {g : Hom b a} → is-prop (Inverses f g)
Inverses-are-prop x y i .Inverses.invl = Hom-set _ _ _ _ (x .invl) (y .invl) i
Inverses-are-prop x y i .Inverses.invr = Hom-set _ _ _ _ (x .invr) (y .invr) i

is-invertible-is-prop : ∀ {f : Hom a b} → is-prop (is-invertible f)
is-invertible-is-prop {a = a} {b = b} {f = f} g h = p where
  module g = is-invertible g
  module h = is-invertible h

  g≡h : g.inv ≡ h.inv
  g≡h =
    g.inv             ≡⟨ sym (idr _) ∙ ap₂ _∘_ refl (sym h.invl) ⟩
    g.inv ∘ f ∘ h.inv ≡⟨ assoc _ _ _ ·· ap₂ _∘_ g.invr refl ·· idl _ ⟩
    h.inv             ∎

  p : g ≡ h
  p i .is-invertible.inv = g≡h i
  p i .is-invertible.inverses =
    is-prop→pathp (λ i → Inverses-are-prop {g = g≡h i}) g.inverses h.inverses i
```

We note that the identity morphism is always iso, and that isos compose:

```agda
id-invertible : ∀ {a} → is-invertible (id {a})
id-invertible .is-invertible.inv = id
id-invertible .is-invertible.inverses .invl = idl id
id-invertible .is-invertible.inverses .invr = idl id

id-iso : a ≅ a
id-iso .to = id
id-iso .from = id
id-iso .inverses .invl = idl id
id-iso .inverses .invr = idl id

Isomorphism = _≅_

Inverses-∘ : {f : Hom a b} {f⁻¹ : Hom b a} {g : Hom b c} {g⁻¹ : Hom c b}
           → Inverses f f⁻¹ → Inverses g g⁻¹ → Inverses (g ∘ f) (f⁻¹ ∘ g⁻¹)
Inverses-∘ {f = f} {f⁻¹} {g} {g⁻¹} finv ginv = record { invl = l ; invr = r } where
  module finv = Inverses finv
  module ginv = Inverses ginv

  abstract
    l : (g ∘ f) ∘ f⁻¹ ∘ g⁻¹ ≡ id
    l = (g ∘ f) ∘ f⁻¹ ∘ g⁻¹ ≡⟨ cat! C ⟩
        g ∘ (f ∘ f⁻¹) ∘ g⁻¹ ≡⟨ (λ i → g ∘ finv.invl i ∘ g⁻¹) ⟩
        g ∘ id ∘ g⁻¹        ≡⟨ cat! C ⟩
        g ∘ g⁻¹             ≡⟨ ginv.invl ⟩
        id                  ∎

    r : (f⁻¹ ∘ g⁻¹) ∘ g ∘ f ≡ id
    r = (f⁻¹ ∘ g⁻¹) ∘ g ∘ f ≡⟨ cat! C ⟩
        f⁻¹ ∘ (g⁻¹ ∘ g) ∘ f ≡⟨ (λ i → f⁻¹ ∘ ginv.invr i ∘ f) ⟩
        f⁻¹ ∘ id ∘ f        ≡⟨ cat! C ⟩
        f⁻¹ ∘ f             ≡⟨ finv.invr ⟩
        id                  ∎

_∘Iso_ : a ≅ b → b ≅ c → a ≅ c
(f ∘Iso g) .to = g .to ∘ f .to
(f ∘Iso g) .from = f .from ∘ g .from
(f ∘Iso g) .inverses = Inverses-∘ (f .inverses) (g .inverses)

infixr 40 _∘Iso_
infixr 41 _Iso⁻¹

invertible-∘
  : ∀ {f : Hom b c} {g : Hom a b}
  → is-invertible f → is-invertible g
  → is-invertible (f ∘ g)
invertible-∘ f-inv g-inv = record
  { inv = g-inv.inv ∘ f-inv.inv
  ; inverses = Inverses-∘ g-inv.inverses f-inv.inverses
  }
  where
    module f-inv = is-invertible f-inv
    module g-inv = is-invertible g-inv

_invertible⁻¹
  : ∀ {f : Hom a b} → (f-inv : is-invertible f)
  → is-invertible (is-invertible.inv f-inv)
_invertible⁻¹ {f = f} f-inv .is-invertible.inv = f
_invertible⁻¹ f-inv .is-invertible.inverses .invl =
  is-invertible.invr f-inv
_invertible⁻¹ f-inv .is-invertible.inverses .invr =
  is-invertible.invl f-inv

_Iso⁻¹ : a ≅ b → b ≅ a
(f Iso⁻¹) .to = f .from
(f Iso⁻¹) .from = f .to
(f Iso⁻¹) .inverses .invl = f .inverses .invr
(f Iso⁻¹) .inverses .invr = f .inverses .invl
```

<!--
```agda
make-inverses : {f : Hom a b} {g : Hom b a} → f ∘ g ≡ id → g ∘ f ≡ id → Inverses f g
make-inverses p q .invl = p
make-inverses p q .invr = q

make-invertible : {f : Hom a b} → (g : Hom b a) → f ∘ g ≡ id → g ∘ f ≡ id → is-invertible f
make-invertible g p q .is-invertible.inv = g
make-invertible g p q .is-invertible.inverses .invl = p
make-invertible g p q .is-invertible.inverses .invr = q

make-iso : (f : Hom a b) (g : Hom b a) → f ∘ g ≡ id → g ∘ f ≡ id → a ≅ b
make-iso f g p q ._≅_.to = f
make-iso f g p q ._≅_.from = g
make-iso f g p q ._≅_.inverses .Inverses.invl = p
make-iso f g p q ._≅_.inverses .Inverses.invr = q

inverses→invertible : ∀ {f : Hom a b} {g : Hom b a} → Inverses f g → is-invertible f
inverses→invertible x .is-invertible.inv = _
inverses→invertible x .is-invertible.inverses = x

_≅⟨_⟩_ : ∀ (x : Ob) {y z} → x ≅ y → y ≅ z → x ≅ z
x ≅⟨ p ⟩ q = p ∘Iso q

_≅˘⟨_⟩_ : ∀ (x : Ob) {y z} → y ≅ x → y ≅ z → x ≅ z
x ≅˘⟨ p ⟩ q = (p Iso⁻¹) ∘Iso q

_≅∎ : (x : Ob) → x ≅ x
x ≅∎ = id-iso

infixr 2 _≅⟨_⟩_ _≅˘⟨_⟩_
infix  3 _≅∎

invertible→iso : (f : Hom a b) → is-invertible f → a ≅ b
invertible→iso f x =
  record
    { to       = f
    ; from     = x .is-invertible.inv
    ; inverses = x .is-invertible.inverses
    }

is-invertible-inverse
  : {f : Hom a b} (g : is-invertible f) → is-invertible (g .is-invertible.inv)
is-invertible-inverse g =
  record { inv = _ ; inverses = record { invl = invr g ; invr = invl g } }
  where open Inverses (g .is-invertible.inverses)

iso→invertible : (i : a ≅ b) → is-invertible (i ._≅_.to)
iso→invertible i = record { inv = i ._≅_.from ; inverses = i ._≅_.inverses }

private
  ≅-pathp-internal
    : (p : a ≡ c) (q : b ≡ d)
    → {f : a ≅ b} {g : c ≅ d}
    → PathP (λ i → Hom (p i) (q i)) (f ._≅_.to) (g ._≅_.to)
    → PathP (λ i → Hom (q i) (p i)) (f ._≅_.from) (g ._≅_.from)
    → PathP (λ i → p i ≅ q i) f g
  ≅-pathp-internal p q r s i .to = r i
  ≅-pathp-internal p q r s i .from = s i
  ≅-pathp-internal p q {f} {g} r s i .inverses =
    is-prop→pathp (λ j → Inverses-are-prop {f = r j} {g = s j})
      (f .inverses) (g .inverses) i

abstract
  inverse-unique
    : {x y : Ob} (p : x ≡ y) {b d : Ob} (q : b ≡ d) (f : x ≅ b) (g : y ≅ d)
    → PathP (λ i → Hom (p i) (q i)) (f .to) (g .to)
    → PathP (λ i → Hom (q i) (p i)) (f .from) (g .from)
  inverse-unique =
    J' (λ a c p → ∀ {b d} (q : b ≡ d) (f : a ≅ b) (g : c ≅ d)
      → PathP (λ i → Hom (p i) (q i)) (f .to) (g .to)
      → PathP (λ i → Hom (q i) (p i)) (f .from) (g .from))
      λ x → J' (λ b d q → (f : x ≅ b) (g : x ≅ d)
                → PathP (λ i → Hom x (q i)) (f .to) (g .to)
                → PathP (λ i → Hom (q i) x) (f .from) (g .from))
            λ y f g p →
              f .from                     ≡˘⟨ ap (f .from ∘_) (g .invl) ∙ idr _ ⟩
              f .from ∘ g .to ∘ g .from   ≡⟨ assoc _ _ _ ⟩
              (f .from ∘ g .to) ∘ g .from ≡⟨ ap (_∘ g .from) (ap (f .from ∘_) (sym p) ∙ f .invr) ∙ idl _ ⟩
              g .from                     ∎

≅-pathp
  : (p : a ≡ c) (q : b ≡ d) {f : a ≅ b} {g : c ≅ d}
  → PathP (λ i → Hom (p i) (q i)) (f ._≅_.to) (g ._≅_.to)
  → PathP (λ i → p i ≅ q i) f g
≅-pathp p q {f = f} {g = g} r = ≅-pathp-internal p q r (inverse-unique p q f g r)

≅-pathp-from
  : (p : a ≡ c) (q : b ≡ d) {f : a ≅ b} {g : c ≅ d}
  → PathP (λ i → Hom (q i) (p i)) (f ._≅_.from) (g ._≅_.from)
  → PathP (λ i → p i ≅ q i) f g
≅-pathp-from p q {f = f} {g = g} r = ≅-pathp-internal p q (inverse-unique q p (f Iso⁻¹) (g Iso⁻¹) r) r

≅-path : {f g : a ≅ b} → f ._≅_.to ≡ g ._≅_.to → f ≡ g
≅-path = ≅-pathp refl refl

≅-path-from : {f g : a ≅ b} → f ._≅_.from ≡ g ._≅_.from → f ≡ g
≅-path-from = ≅-pathp-from refl refl

≅-is-contr : (∀ {a b} → is-contr (Hom a b)) → is-contr (a ≅ b)
≅-is-contr hom-contr .centre =
  make-iso (hom-contr .centre) (hom-contr .centre)
    (is-contr→is-prop hom-contr _ _)
    (is-contr→is-prop hom-contr _ _)
≅-is-contr hom-contr .paths f = ≅-path (hom-contr .paths (f .to))

≅-is-prop : (∀ {a b} → is-prop (Hom a b)) → is-prop (a ≅ b)
≅-is-prop hom-prop f g = ≅-path (hom-prop (f .to) (g .to))

↪-pathp
  : {a : I → Ob} {b : I → Ob} {f : a i0 ↪ b i0} {g : a i1 ↪ b i1}
  → PathP (λ i → Hom (a i) (b i)) (f .mor) (g .mor)
  → PathP (λ i → a i ↪ b i) f g
↪-pathp {a = a} {b} {f} {g} pa = go where
  go : PathP (λ i → a i ↪ b i) f g
  go i .mor = pa i
  go i .monic {c = c} =
    is-prop→pathp
      (λ i → Π-is-hlevel {A = Hom c (a i)} 1
       λ g → Π-is-hlevel {A = Hom c (a i)} 1
       λ h → fun-is-hlevel {A = pa i ∘ g ≡ pa i ∘ h} 1
              (Hom-set c (a i) g h))
      (f .monic)
      (g .monic)
      i

↠-pathp
  : {a : I → Ob} {b : I → Ob} {f : a i0 ↠ b i0} {g : a i1 ↠ b i1}
  → PathP (λ i → Hom (a i) (b i)) (f .mor) (g .mor)
  → PathP (λ i → a i ↠ b i) f g
↠-pathp {a = a} {b} {f} {g} pa = go where
  go : PathP (λ i → a i ↠ b i) f g
  go i .mor = pa i
  go i .epic {c = c} =
    is-prop→pathp
      (λ i → Π-is-hlevel {A = Hom (b i) c} 1
       λ g → Π-is-hlevel {A = Hom (b i) c} 1
       λ h → fun-is-hlevel {A = g ∘ pa i ≡ h ∘ pa i} 1
              (Hom-set (b i) c g h))
      (f .epic)
      (g .epic)
      i
```
-->

Isomorphisms enjoy a **2-out-of-3** property: if any 2 of $f$, $g$, and
$f \circ g$ are isomorphisms, then so is the third.

```agda
inverses-cancell
  : ∀ {f : Hom b c} {g : Hom a b} {g⁻¹ : Hom b a} {fg⁻¹ : Hom c a}
  → Inverses g g⁻¹ → Inverses (f ∘ g) fg⁻¹
  → Inverses f (g ∘ fg⁻¹)

inverses-cancelr
  : ∀ {f : Hom b c} {f⁻¹ : Hom c b} {g : Hom a b} {fg⁻¹ : Hom c a}
  → Inverses f f⁻¹ → Inverses (f ∘ g) fg⁻¹
  → Inverses g (fg⁻¹ ∘ f)

invertible-cancell
  : ∀ {f : Hom b c} {g : Hom a b}
  → is-invertible g → is-invertible (f ∘ g)
  → is-invertible f

invertible-cancelr
  : ∀ {f : Hom b c} {g : Hom a b}
  → is-invertible f → is-invertible (f ∘ g)
  → is-invertible g
```

<details>
<summary>We are early into our bootstrapping process for category theory,
so the proofs of these lemmas are quite low-level, and thus omitted.
</summary>

```agda
inverses-cancell g-inv fg-inv .invl =
  assoc _ _ _ ∙ fg-inv .invl
inverses-cancell g-inv fg-inv .invr =
  sym (idr _)
  ∙ ap₂ _∘_ refl (sym (g-inv .invl))
  ∙ assoc _ _ _
  ∙ ap₂ _∘_
    (sym (assoc _ _ _)
    ∙ sym (assoc _ _ _)
    ∙ ap₂ _∘_ refl (fg-inv .invr)
    ∙ idr _)
    refl
  ∙ g-inv .invl

inverses-cancelr f-inv fg-inv .invl =
  sym (idl _)
  ∙ ap₂ _∘_ (sym (f-inv .invr)) refl
  ∙ sym (assoc _ _ _)
  ∙ ap₂ _∘_ refl
    (assoc _ _ _
    ∙ assoc _ _ _
    ∙ ap₂ _∘_ (fg-inv .invl) refl
    ∙ idl _)
  ∙ f-inv .invr
inverses-cancelr f-inv fg-inv .invr =
  sym (assoc _ _ _) ∙ fg-inv .invr

invertible-cancell g-inv fg-inv =
  inverses→invertible $
  inverses-cancell
    (g-inv .is-invertible.inverses)
    (fg-inv .is-invertible.inverses)

invertible-cancelr f-inv fg-inv =
  inverses→invertible $
  inverses-cancelr
    (f-inv .is-invertible.inverses)
    (fg-inv .is-invertible.inverses)
```
</details>

We also note that invertible morphisms are both epic and monic.

```agda
invertible→monic : is-invertible f → is-monic f
invertible→monic {f = f} invert g h p =
  g             ≡˘⟨ idl g ⟩
  id ∘ g        ≡˘⟨ ap (_∘ g) (is-invertible.invr invert) ⟩
  (inv ∘ f) ∘ g ≡˘⟨ assoc inv f g ⟩
  inv ∘ (f ∘ g) ≡⟨ ap (inv ∘_) p ⟩
  inv ∘ (f ∘ h) ≡⟨ assoc inv f h ⟩
  (inv ∘ f) ∘ h ≡⟨ ap (_∘ h) (is-invertible.invr invert) ⟩
  id ∘ h        ≡⟨ idl h ⟩
  h ∎
  where
    open is-invertible invert

invertible→epic : is-invertible f → is-epic f
invertible→epic {f = f} invert g h p =
  g             ≡˘⟨ idr g ⟩
  g ∘ id        ≡˘⟨ ap (g ∘_) (is-invertible.invl invert) ⟩
  g ∘ (f ∘ inv) ≡⟨ assoc g f inv ⟩
  (g ∘ f) ∘ inv ≡⟨ ap (_∘ inv) p ⟩
  (h ∘ f) ∘ inv ≡˘⟨ assoc h f inv ⟩
  h ∘ (f ∘ inv) ≡⟨ ap (h  ∘_) (is-invertible.invl invert) ⟩
  h ∘ id        ≡⟨ idr h ⟩
  h ∎
  where
    open is-invertible invert

iso→monic : (f : a ≅ b) → is-monic (f .to)
iso→monic f = invertible→monic (iso→invertible f)

iso→epic : (f : a ≅ b) → is-epic (f .to)
iso→epic f = invertible→epic (iso→invertible f)
```

Furthermore, isomorphisms also yield section/retraction pairs.

```agda
inverses→to-has-section
  : ∀ {f : Hom a b} {g : Hom b a}
  → Inverses f g → has-section f
inverses→to-has-section {g = g} inv .section = g
inverses→to-has-section inv .is-section = Inverses.invl inv

inverses→from-has-section
  : ∀ {f : Hom a b} {g : Hom b a}
  → Inverses f g → has-section g
inverses→from-has-section {f = f} inv .section = f
inverses→from-has-section inv .is-section = Inverses.invr inv

inverses→to-has-retract
  : ∀ {f : Hom a b} {g : Hom b a}
  → Inverses f g → has-retract f
inverses→to-has-retract {g = g} inv .retract = g
inverses→to-has-retract inv .is-retract = Inverses.invr inv

inverses→from-has-retract
  : ∀ {f : Hom a b} {g : Hom b a}
  → Inverses f g → has-retract g
inverses→from-has-retract {f = f} inv .retract = f
inverses→from-has-retract inv .is-retract = Inverses.invl inv

iso→to-has-section : (f : a ≅ b) → has-section (f .to)
iso→to-has-section f .section = f .from
iso→to-has-section f .is-section = f .invl

iso→from-has-section : (f : a ≅ b) → has-section (f .from)
iso→from-has-section f .section = f .to
iso→from-has-section f .is-section = f .invr

iso→to-has-retract : (f : a ≅ b) → has-retract (f .to)
iso→to-has-retract f .retract = f .from
iso→to-has-retract f .is-retract = f .invr

iso→from-has-retract : (f : a ≅ b) → has-retract (f .from)
iso→from-has-retract f .retract = f .to
iso→from-has-retract f .is-retract = f .invl
```

Using our lemmas about section/retraction pairs from before, we
can show that if $f$ is a monic retract, then $f$ is an
iso.

```agda
retract-of+monic→inverses
  : ∀ {a b} {s : Hom b a} {r : Hom a b}
  → r retract-of s
  → is-monic r
  → Inverses r s
retract-of+monic→inverses ret monic .invl = ret
retract-of+monic→inverses ret monic .invr =
  retract-of+monic→section-of ret monic
```

We also have a dual result for sections and epis.

```agda
section-of+epic→inverses
  : ∀ {a b} {s : Hom b a} {r : Hom a b}
  → s retract-of r
  → is-epic r
  → Inverses r s
section-of+epic→inverses sect epic .invl =
  section-of+epic→retract-of sect epic
section-of+epic→inverses sect epic .invr = sect
```

<!--
```agda
has-retract+epic→invertible
  : ∀ {a b} {f : Hom a b}
  → has-retract f
  → is-epic f
  → is-invertible f
has-retract+epic→invertible f-ret epic .is-invertible.inv =
  f-ret .retract
has-retract+epic→invertible f-ret epic .is-invertible.inverses =
  section-of+epic→inverses (f-ret .is-retract) epic

has-section+monic→invertible
  : ∀ {a b} {f : Hom a b}
  → has-section f
  → is-monic f
  → is-invertible f
has-section+monic→invertible f-sect monic .is-invertible.inv =
  f-sect .section
has-section+monic→invertible f-sect monic .is-invertible.inverses =
  retract-of+monic→inverses (f-sect .is-section) monic
```
-->

Hom-sets between isomorphic objects are equivalent. Crucially, this
allows us to use [[univalence]] to transport properties between
hom-sets.

```agda
iso→hom-equiv
  : ∀ {a a' b b'} → a ≅ a' → b ≅ b'
  → Hom a b ≃ Hom a' b'
iso→hom-equiv f g = Iso→Equiv $
  (λ h → g .to ∘ h ∘ f .from) ,
  iso (λ h → g .from ∘ h ∘ f .to )
    (λ h →
      g .to ∘ (g .from ∘ h ∘ f .to) ∘ f .from   ≡⟨ cat! C ⟩
      (g .to ∘ g .from) ∘ h ∘ (f .to ∘ f .from) ≡⟨ ap₂ (λ l r → l ∘ h ∘ r) (g .invl) (f .invl) ⟩
      id ∘ h ∘ id                               ≡⟨ cat! C ⟩
      h ∎)
    (λ h →
      g .from ∘ (g .to ∘ h ∘ f .from) ∘ f .to   ≡⟨ cat! C ⟩
      (g .from ∘ g .to) ∘ h ∘ (f .from ∘ f .to) ≡⟨ ap₂ (λ l r → l ∘ h ∘ r) (g .invr) (f .invr) ⟩
      id ∘ h ∘ id                               ≡⟨ cat! C ⟩
      h ∎)
```

<!--
```agda
-- Optimized versions of Iso→Hom-equiv which yield nicer results
-- if only one isomorphism is needed.
dom-iso→hom-equiv
  : ∀ {a a' b} → a ≅ a'
  → Hom a b ≃ Hom a' b
dom-iso→hom-equiv f = Iso→Equiv $
  (λ h → h ∘ f .from) ,
  iso (λ h → h ∘ f .to )
    (λ h →
      (h ∘ f .to) ∘ f .from ≡⟨ sym (assoc _ _ _) ⟩
      h ∘ (f .to ∘ f .from) ≡⟨ ap (h ∘_) (f .invl) ⟩
      h ∘ id                ≡⟨ idr _ ⟩
      h ∎)
    (λ h →
      (h ∘ f .from) ∘ f .to ≡⟨ sym (assoc _ _ _) ⟩
      h ∘ (f .from ∘ f .to) ≡⟨ ap (h ∘_) (f .invr) ⟩
      h ∘ id                ≡⟨ idr _ ⟩
      h ∎)

cod-iso→Hom-equiv
  : ∀ {a b b'} → b ≅ b'
  → Hom a b ≃ Hom a b'
cod-iso→Hom-equiv f = Iso→Equiv $
  (λ h → f .to ∘ h) ,
  iso (λ h → f .from ∘ h)
    (λ h →
      f .to ∘ f .from ∘ h   ≡⟨ assoc _ _ _ ⟩
      (f .to ∘ f .from) ∘ h ≡⟨ ap (_∘ h) (f .invl) ⟩
      id ∘ h                ≡⟨ idl _ ⟩
      h ∎)
    (λ h →
      f .from ∘ f .to ∘ h   ≡⟨ assoc _ _ _ ⟩
      (f .from ∘ f .to) ∘ h ≡⟨ ap (_∘ h) (f .invr) ⟩
      id ∘ h                ≡⟨ idl _ ⟩
      h ∎)
```
-->

If $f$ is invertible, then both pre and post-composition with $f$ are
equivalences.

```agda
invertible-precomp-equiv
  : ∀ {a b c} {f : Hom a b}
  → is-invertible f
  → is-equiv {A = Hom b c} (_∘ f)
invertible-precomp-equiv {f = f} f-inv = is-iso→is-equiv $
  iso (λ h → h ∘ f-inv.inv)
    (λ h → sym (assoc _ _ _) ·· ap (h ∘_) f-inv.invr ·· idr h)
    (λ h → sym (assoc _ _ _) ·· ap (h ∘_) f-inv.invl ·· idr h)
  where module f-inv = is-invertible f-inv

invertible-postcomp-equiv
  : ∀ {a b c} {f : Hom b c}
  → is-invertible f
  → is-equiv {A = Hom a b} (f ∘_)
invertible-postcomp-equiv {f = f} f-inv = is-iso→is-equiv $
  iso (λ h → f-inv.inv ∘ h)
    (λ h → assoc _ _ _ ·· ap (_∘ h) f-inv.invl ·· idl h)
    (λ h → assoc _ _ _ ·· ap (_∘ h) f-inv.invr ·· idl h)
  where module f-inv = is-invertible f-inv
```
