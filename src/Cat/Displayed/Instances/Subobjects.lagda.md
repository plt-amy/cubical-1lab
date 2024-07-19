<!--
```agda
open import Cat.Diagram.Pullback.Properties
open import Cat.Displayed.Cocartesian.Weak
open import Cat.Displayed.Cocartesian
open import Cat.Displayed.Univalence
open import Cat.Displayed.Cartesian
open import Cat.Instances.Functor
open import Cat.Diagram.Pullback
open import Cat.Diagram.Product
open import Cat.Displayed.Fibre
open import Cat.Displayed.Base
open import Cat.Diagram.Image
open import Cat.Prelude

import Cat.Reasoning as Cr
```
-->

```agda
module Cat.Displayed.Instances.Subobjects
  {o ℓ} (B : Precategory o ℓ)
  where
```

<!--
```agda
open Cr B
open Displayed
```
-->

# The fibration of subobjects {defines="poset-of-subobjects subobject"}

Given a base category $\cB$, we can define the [[displayed category]] of
_subobjects_ over $\cB$. This is, in essence, a restriction of the
[[codomain fibration]] of $\cB$, but with our attention restricted to
the monomorphisms $a \mono b$ rather than arbitrary maps $a \to b$.

```agda
record Subobject (y : Ob) : Type (o ⊔ ℓ) where
  no-eta-equality
  field
    {domain} : Ob
    map   : Hom domain y
    monic : is-monic map

open Subobject public
```

To make formalisation smoother, we define our own version of displayed
morphisms in the subobject fibration, rather than reusing those of the
fundamental self-indexing. The reason for this is quite technical: the
type of maps in the self-indexing is only dependent (the domains and)
the _morphisms_ being considered, meaning nothing constrains the proofs
that these are monomorphisms, causing unification to fail at the
determining the full `Subobject`{.Agda}s involved.

```agda
record ≤-over {x y} (f : Hom x y) (a : Subobject x) (b : Subobject y) : Type ℓ where
  no-eta-equality
  field
    map : Hom (a .domain) (b .domain)
    sq : f ∘ Subobject.map a ≡ Subobject.map b ∘ map

open ≤-over public
```

We will denote the type of maps $x' \to_f y'$ in the subobject fibration
by $x' \le_f y'$, since there is at most one such map: if $g, h$ satisfy
$y'g = fx' = y'h$, then, since $y'$ is a mono, $g = h$.

<!--
```agda
≤-over-is-prop
  : ∀ {x y} {f : Hom x y} {a : Subobject x} {b : Subobject y}
  → (p q : ≤-over f a b)
  → p ≡ q
≤-over-is-prop {f = f} {a} {b} p q = path where
  maps : p .map ≡ q .map
  maps = b .monic (p .map) (q .map) (sym (p .sq) ∙ q .sq)

  path : p ≡ q
  path i .map = maps i
  path i .sq = is-prop→pathp (λ i → Hom-set _ _ (f ∘ a .map) (b .map ∘ maps i)) (p .sq) (q .sq) i

instance
  H-Level-≤-over
    : ∀ {x y} {f : Hom x y} {a : Subobject x} {b : Subobject y} {n}
    → H-Level (≤-over f a b) (suc n)
  H-Level-≤-over = prop-instance ≤-over-is-prop
```
-->

Setting up the displayed category is now nothing more than routine
verification: the identity map satisfies $\id a = a \id$, and
commutative squares can be pasted together.

```agda
Subobjects : Displayed B (o ⊔ ℓ) ℓ
Subobjects .Ob[_] y = Subobject y
Subobjects .Hom[_]  = ≤-over
Subobjects .Hom[_]-set f a b = hlevel 2

Subobjects .id' .map = id
Subobjects .id' .sq  = id-comm-sym

Subobjects ._∘'_ α β .map = α .map ∘ β .map
Subobjects ._∘'_ α β .sq  = pullr (β .sq) ∙ extendl (α .sq)
```

<!--
```agda
Subobjects .idr' _       = prop!
Subobjects .idl' _       = prop!
Subobjects .assoc' _ _ _ = prop!

open is-weak-cocartesian-fibration
open Weak-cocartesian-lift
open Cartesian-fibration
open is-weak-cocartesian
open Cartesian-lift
open is-cartesian
open Pullback
```
-->

## As a fibration

By exactly the same construction as [for the fundamental
self-indexing][codomain], if $\cB$ has pullbacks, the displayed category
we have built is actually a fibration. The construction is slightly
simpler now that we have no need to worry about uniqueness, but we can
remind ourselves of the universal property:

[codomain]: Cat.Displayed.Instances.Slice.html#as-a-fibration

~~~{.quiver}
\[\begin{tikzcd}
  \textcolor{rgb,255:red,214;green,92;blue,92}{u'} \\
  & \textcolor{rgb,255:red,92;green,92;blue,214}{x \times_y y'} && {y'} \\
  \textcolor{rgb,255:red,214;green,92;blue,92}{u} \\
  & x && y
  \arrow[hook, from=2-4, to=4-4]
  \arrow["f"', from=4-2, to=4-4]
  \arrow[color={rgb,255:red,92;green,92;blue,214}, hook, from=2-2, to=4-2]
  \arrow[color={rgb,255:red,92;green,92;blue,214}, from=2-2, to=2-4]
  \arrow["m"', color={rgb,255:red,214;green,92;blue,92}, from=3-1, to=4-2]
  \arrow[color={rgb,255:red,214;green,92;blue,92}, hook, from=1-1, to=3-1]
  \arrow["{\exists!}"', color={rgb,255:red,214;green,92;blue,92}, dashed, from=1-1, to=2-2]
  \arrow["h", color={rgb,255:red,214;green,92;blue,92}, curve={height=-12pt}, from=1-1, to=2-4]
\end{tikzcd}\]
~~~

On the first stage, we are given the data in black: we can complete an
open span $y' \mono y \xot{f} x$ to a Cartesian square (in blue) by
pulling $y'$ back along $f$; this base change remains a monomorphism.
Now given the data in red, we verify that the dashed arrow exists, which
is enough for its uniqueness.

```agda
-- The blue square:
pullback-subobject
  : has-pullbacks B
  → ∀ {X Y} (h : Hom X Y) (g : Subobject Y)
  → Subobject X
pullback-subobject pb h g .domain = pb h (g .map) .apex
pullback-subobject pb h g .map = pb h (g .map) .p₁
pullback-subobject pb h g .monic = is-monic→pullback-is-monic
  (g .monic) (rotate-pullback (pb h (g .map) .has-is-pb))

Subobject-fibration
  : has-pullbacks B
  → Cartesian-fibration Subobjects
Subobject-fibration pb .has-lift f y' = l where
  l : Cartesian-lift Subobjects f y'

  l .x' = pullback-subobject pb f y'
  l .lifting .map = pb f (y' .map) .p₂
  l .lifting .sq  = pb f (y' .map) .square

  -- The dashed red arrow:
  l .cartesian .universal {u' = u'} m h' = λ where
    .map → pb f (y' .map) .universal (pushr refl ∙ h' .sq)
    .sq  → sym (pb f (y' .map) .p₁∘universal)
  l .cartesian .commutes _ _ = prop!
  l .cartesian .unique _ _   = prop!
```

## As a (weak) cocartesian fibration

If $\cB$ has an [[image factorisation]] for every morphism, then its
fibration of subobjects is a weak cocartesian fibration. By a general
fact, if $\cB$ also has pullbacks, then $\Sub(-)$ is a cocartesian
fibration.

```agda
Subobject-weak-opfibration
  : (∀ {x y} (f : Hom x y) → Image B f)
  → is-weak-cocartesian-fibration Subobjects
Subobject-weak-opfibration ims .weak-lift f x' = l where
  module im = Image B (ims (f ∘ x' .map))
```

To understand this result, we remind ourselves of the universal property
of an image factorisation for $f : a \to b$: It is the initial subobject
through with $f$ factors. That is to say, if $m : \Sub(b)$ is another
subobject, and $f = me$ for some map $e : a \to m$, then $m \le \im f$.
Summarised diagrammatically, the universal property of an image
factorisation looks like a kite:

~~~{.quiver}
\[\begin{tikzcd}
  a && {\im f} && m \\
  \\
  && b
  \arrow[from=1-1, to=1-3]
  \arrow[hook, from=1-3, to=3-3]
  \arrow["{\exists!}"', dashed, from=1-3, to=1-5]
  \arrow[hook, from=1-5, to=3-3]
  \arrow["e"{description}, curve={height=-18pt}, from=1-1, to=1-5]
  \arrow["f"{description}, from=1-1, to=3-3]
\end{tikzcd}\]
~~~

Now compare this with the universal property required of a weak
co-cartesian lift:

~~~{.quiver}
\[\begin{tikzcd}
  {x'} && {f_!x'} && {u'} \\
  \\
  x && y
  \arrow[hook, from=1-1, to=3-1]
  \arrow["f"', from=3-1, to=3-3]
  \arrow[hook, from=1-3, to=3-3]
  \arrow[hook, from=1-5, to=3-3]
  \arrow[dashed, from=1-1, to=1-3]
  \arrow["h", curve={height=-18pt}, from=1-1, to=1-5]
\end{tikzcd}\]
~~~

By smooshing the corner $x' \mono x \to y$ together (i.e., composing
$x'$ and $f$), we see that this is exactly the kite-shaped universal
property of $\im fx'$.

```agda
  l : Weak-cocartesian-lift Subobjects f x'
  l .y' .domain = im.Im
  l .y' .map    = im.Im→codomain
  l .y' .monic  = im.Im→codomain-is-M

  l .lifting .map = im.corestrict
  l .lifting .sq  = sym im.image-factors

  l .weak-cocartesian .universal {x' = y'} h .map = im.universal _ (y' .monic) (h .map) (sym (h .sq))
  l .weak-cocartesian .universal h .sq = idl _ ∙ sym im.universal-factors

  l .weak-cocartesian .commutes g' = prop!
  l .weak-cocartesian .unique _ _  = prop!
```

The aforementioned general fact says that any cartesian and weak
cocartesian fibration must actually be a full opfibration.

```agda
Subobject-opfibration
  : (∀ {x y} (f : Hom x y) → Image B f)
  → (pb : has-pullbacks B)
  → Cocartesian-fibration Subobjects
Subobject-opfibration images pb = cartesian+weak-opfibration→opfibration _
  (Subobject-fibration pb)
  (Subobject-weak-opfibration images)
```

## Subobjects over a base

We define the category $\Sub(y)$ of subobjects _of $y$_ as a fibre of
the subobject fibration. However, we use a purpose-built transport
function to cut down on the number of coherences required to work with
$\Sub(y)$ at use-sites.

```agda
Sub : Ob → Precategory (o ⊔ ℓ) ℓ
Sub y = Fibre' Subobjects y re coh where
  re : ∀ {a b} → ≤-over (id ∘ id) a b → ≤-over id a b
  re x .map = x .map
  re x .sq  = ap₂ _∘_ (introl refl) refl ∙ x .sq

  abstract
    coh : ∀ {a b} (f : ≤-over (id ∘ id) a b) → re f ≡ transport (λ i → ≤-over (idl id i) a b) f
    coh f = prop!

module Sub {y} = Cr (Sub y)
```

<!--
```agda
_≤ₘ_ : ∀ {y} (a b : Subobject y) → Type _
_≤ₘ_ = ≤-over id

≤ₘ→mono : ∀ {y} {a b : Subobject y} → a ≤ₘ b → a .domain ↪ b .domain
≤ₘ→mono x .mor = x .map
≤ₘ→mono {a = a} x .monic g h α = a .monic g h $
  a .map ∘ g      ≡⟨ ap (_∘ g) (introl refl ∙ x .sq) ∙ pullr refl ⟩
  _ ∘ x .map ∘ g  ≡⟨ ap₂ _∘_ refl α ⟩
  _ ∘ x .map ∘ h  ≡⟨ pulll (sym (x .sq) ∙ idl _) ⟩
  a .map ∘ h      ∎

cutₛ : ∀ {x y} {f : Hom x y} → is-monic f → Subobject y
cutₛ x .domain = _
cutₛ x .map    = _
cutₛ x .monic  = x

Sub-antisym
  : ∀ {y} {a b : Subobject y}
  → a ≤ₘ b
  → b ≤ₘ a
  → a Sub.≅ b
Sub-antisym f g = Sub.make-iso f g prop! prop!

Sub-path
  : ∀ {y} {a b : Subobject y}
  → (p : a .domain ≡ b .domain)
  → PathP (λ i → Hom (p i) y) (a .map) (b .map)
  → a ≡ b
Sub-path p q i .domain = p i
Sub-path p q i .map = q i
Sub-path {a = a} {b = b} p q i .monic {c} =
  is-prop→pathp (λ i → Π-is-hlevel³ 1 λ (g h : Hom c (p i)) (_ : q i ∘ g ≡ q i ∘ h) → Hom-set _ _ g h)
    (a .monic) (b .monic) i
```
-->

## Fibrewise cartesian structure

Since products in slice categories are given by pullbacks, and pullbacks
preserve monomorphisms, if $\cB$ has pullbacks, then $\Sub(y)$ has
products, regardless of what $y$ is.

```agda
Sub-products
  : ∀ {y}
  → has-pullbacks B
  → has-products (Sub y)
Sub-products {y} pb a b = prod where
  it = pb (a .map) (b .map)

  prod : Product (Sub y) a b
  prod .Product.apex .domain = it .apex
  prod .Product.apex .map = a .map ∘ it .p₁
  prod .Product.apex .monic = monic-∘
    (a .monic)
    (is-monic→pullback-is-monic (b .monic) (rotate-pullback (it .has-is-pb)))

  prod .Product.π₁ .map = it .p₁
  prod .Product.π₁ .sq  = idl _

  prod .Product.π₂ .map = it .p₂
  prod .Product.π₂ .sq  = idl _ ∙ it .square

  prod .Product.has-is-product .is-product.⟨_,_⟩ q≤a q≤b .map =
    it .Pullback.universal {p₁' = q≤a .map} {p₂' = q≤b .map} (sym (q≤a .sq) ∙ q≤b .sq)
  prod .Product.has-is-product .is-product.⟨_,_⟩ q≤a q≤b .sq =
    idl _ ∙ sym (pullr (it .p₁∘universal) ∙ sym (q≤a .sq) ∙ idl _)
  prod .Product.has-is-product .is-product.π₁∘factor = prop!
  prod .Product.has-is-product .is-product.π₂∘factor = prop!
  prod .Product.has-is-product .is-product.unique _ _ _ = prop!
```

## Univalence

Since identity of $m, n : \Sub(y)$ is given by identity of they
underlying objects and identity-over of the corresponding morphisms, if
$\cB$ is univalent, we can conclude that $\Sub(y)$ is, too. Since
$\Sub(y)$ is always thin, we can summarise the situation by saying that
$\Sub(y)$ is a [[partial order]] if $\cB$ is univalent.

```agda
Sub-is-category : ∀ {y} → is-category B → is-category (Sub y)
Sub-is-category b-cat .to-path {a} {b} x =
  Sub-path
    (b-cat .to-path i)
    (Univalent.Hom-pathp-refll-iso b-cat (sym (x .Sub.from .sq) ∙ idl _))
  where
    i : a .domain ≅ b .domain
    i = make-iso (x .Sub.to .map) (x .Sub.from .map) (ap map (Sub.invl x)) (ap map (Sub.invr x))
Sub-is-category b-cat .to-path-over p =
  Sub.≅-pathp refl _ prop!
```
