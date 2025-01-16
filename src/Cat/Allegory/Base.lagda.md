<!--
```agda
open import Cat.Prelude
```
-->

```agda
module Cat.Allegory.Base where
```

# Allegories {defines="allegory"}

In the same way that a category abstracts over the commonalities of sets
and functions (and thus allows us to study several other mathematical
constructions which are not "set-and-function-like" on the nose using a
common framework), the idea of **allegories** is to abstract over the
commonalities of sets and _relations_. Remarkably, allegories are
defined as category-_like_ structures, as we'll see.

**A note**: In the literature (particularly the nLab), allegories are
defined as a special case of [bicategories]. For ease of presentation,
we have chosen to introduce allegories as their own concept, _a priori_
disconnected from our theory of bicategories.

[bicategories]: Cat.Bi.Base.html

```agda
record Allegory o ℓ ℓ' : Type (lsuc (o ⊔ ℓ ⊔ ℓ')) where
  no-eta-equality
  field cat : Precategory o ℓ
  open Precategory cat public
```

An allegory $\cA$ has an underlying precategory (whose morphisms we
call **relations**), but, more importantly, an ordering relation $f \le
g$ on the relations, which we think of as inclusion in the usual
sense^[This will make more sense when we see the construction of
$\Rel$ below]. When considered with this ordering, $\cA$ must be
a _locally posetal bicategory_: a bicategory $\cA$, with all
Hom-categories being posets.

```agda
  field
    _≤_       : ∀ {x y} → Hom x y → Hom x y → Type ℓ'
    ≤-thin    : ∀ {x y} {f g : Hom x y} → is-prop (f ≤ g)
    ≤-refl    : ∀ {x y} {f : Hom x y} → f ≤ f
    ≤-trans   : ∀ {x y} {f g h : Hom x y} → f ≤ g → g ≤ h → f ≤ h
    ≤-antisym : ∀ {x y} {f g : Hom x y} → f ≤ g → g ≤ f → f ≡ g
```

We want the composition operation to be functorial with respect to the
ordering, which means that $f \le f'$ and $g \le g'$ must imply $fg \le
f'g'$. We have to include this explicitly since we are unpacking the
data of a bicategory.

```agda
    _◆_ : ∀ {w x y} {f f' : Hom x y} {g g' : Hom w x}
        → f ≤ f' → g ≤ g' → (f ∘ g) ≤ (f' ∘ g')
```

<!--
```agda
  infixl 50 _†
  infixr 35 _∩_
  infixr 40 _◆_
  infix 30 _≤_
```
-->

Moreover, $\cA$ must be equipped with an involution on its Hom-sets:
This is the usual opposite of a relation, i.e. $R^o(x,y) = R(y,x)$, and,
as you have noticed, we will write the opposite of $f$ as $f^o$. The
involution must play nice with the ordering, and it must follow the
socks-and-shoes principle[^socks] w.r.t. composition: $(fg)^o = g^of^o$.

[^socks]: This is an allegorical (ha) name to emphasize the reason why
applying the inverse changes the order of multiplication: To put on
socks and shoes, you must first put on your socks, and only then your
shoes; but taking them off, the first thing to come off are the shoes,
and only then can you remove your socks.

```agda
  field
    _†     : ∀ {x y} → Hom x y → Hom y x
    dual   : ∀ {x y} (f : Hom x y) → f † † ≡ f
    dual-∘ : ∀ {w x y} {f : Hom x y} {g : Hom w x} → (f ∘ g) † ≡ g † ∘ f †
    dual-≤ : ∀ {x y} {f g : Hom x y} → f ≤ g → f † ≤ g †
```

The penultimate requirement is that each Hom-poset have binary products,
i.e., intersection of relations. Note that we do not mandate nullary
products, i.e., we do not require an arbitrary allegory to have
"maximal" relations.

```agda
  field
    _∩_    : ∀ {x y} → Hom x y → Hom x y → Hom x y
    ∩-le-l : ∀ {x y} {f g : Hom x y} → f ∩ g ≤ f
    ∩-le-r : ∀ {x y} {f g : Hom x y} → f ∩ g ≤ g
    ∩-univ : ∀ {x y} {f g h : Hom x y} → h ≤ f → h ≤ g → h ≤ f ∩ g
```

Finally, an allegory must satisfy the **modular law**, which we phrase
as the statement below (following Freyd). Allegories thus generalise
modular lattices, in the same way that categories generalise
monoids.^[as a hint: _joins_ in your lattice become composition]

```agda
  field
    modular
      : ∀ {x y z} (f : Hom x y) (g : Hom y z) (h : Hom x z)
      → (g ∘ f) ∩ h ≤ g ∘ (f ∩ (g † ∘ h))
```

## Quick theorems

The first thing we observe about allegories is a Yoneda-type lemma for
relations: Fixing $g$ and $h$, if $f \le g$ is equivalent to $f \le h$
for every $f$, then $g = h$. Needless to say, this holds in any poset
(and really any [[univalent category]]), but we emphasise it _here_, for
allegories, since it will be used to prove important laws.

```agda
module _ {o ℓ ℓ'} (A : Allegory o ℓ ℓ') where
  open Allegory A
  ≤-yoneda
    : ∀ {x y} {g h : Hom x y}
    → (∀ f → f ≤ g → f ≤ h)
    → (∀ f → f ≤ h → f ≤ g)
    → g ≡ h
  ≤-yoneda to fro = ≤-antisym (to _ ≤-refl) (fro _ ≤-refl)
```

Since the duality respects ordering, we can ping-pong $(-)^o$ to get it
out of the way: $f^o \le g$ is exactly the same as $f \le g^o$.

```agda
  dual-≤ₗ : ∀ {x y} {f : Hom x y} {g} → f ≤ g † → f † ≤ g
  dual-≤ₗ {f = f} w = subst (f † ≤_) (dual _) (dual-≤ w)

  dual-≤ᵣ : ∀ {x y} {f : Hom x y} {g} → f † ≤ g → f ≤ g †
  dual-≤ᵣ {f = f} {g} w = subst (_≤ g †) (dual _) (dual-≤ w)
```

As an application of these two quick lemmas and the laws for meets, we
prove that the dual $(f \cap g)^o$ is equivalently given by $f^o \cap
g^o$. We ellaborate the correspondence in one direction: Suppose that $f
\le (f \cap g)^o$. Pinging the dual across, we get $h^o \le (f \cap g)$,
which (by the definition of meet) is equivalent to the conjunction of
$h^o \le f$ and $h^o \le g$. But ponging the dual on either of those, we
get $h \le f^o$ and $h \le g^o$, which mean $h \le f^o \cap g^o$.

```agda
  dual-∩ : ∀ {x y} {f g : Hom x y} → (f ∩ g) † ≡ f † ∩ g †
  dual-∩ {f = f} {g = g} = ≤-yoneda
    (λ h h≤f∩gᵒ →
      let
        hᵒ≤f∩g = dual-≤ₗ h≤f∩gᵒ
        hᵒ≤f = ≤-trans hᵒ≤f∩g ∩-le-l
        hᵒ≤g = ≤-trans hᵒ≤f∩g ∩-le-r
      in ∩-univ (dual-≤ᵣ hᵒ≤f) (dual-≤ᵣ hᵒ≤g))
    (λ h h≤gᵒ∩fᵒ →
      let
        h≤gᵒ = ≤-trans h≤gᵒ∩fᵒ ∩-le-r
        h≤fᵒ = ≤-trans h≤gᵒ∩fᵒ ∩-le-l
      in dual-≤ᵣ (∩-univ (dual-≤ₗ h≤fᵒ) (dual-≤ₗ h≤gᵒ)))

  dual-id : ∀ {x} → id {x = x} † ≡ id
  dual-id = sym (sym (dual id) ∙ ap _† (sym (idl _)) ∙ dual-∘ ∙ ap (_∘ id †) (dual _) ∙ idl _)
```

# The allegory of relations

The allegorical analogue of the category of sets is the **allegory of
relations**, $\Rel$. As usual, every universe level $\kappa$ gives
an allegory $\Rel_\kappa$, whose objects are $\kappa$-small sets,
and morphisms are $\kappa$-small relations. Note that, in the absence of
propositional resizing, $\Rel$'s underlying category is _not_
locally $\kappa$-small: The set of $\kappa$-small propositions lives in
the successor universe, not in $\kappa$.

<!--
```agda
open Allegory
open Precategory
```
-->

Ordering of relations is as one would expect: $R \le S$ is defined to
mean every $R$-related pair $(x, y)$ is also $S$-related. When seen as
subsets of $A \times B$, this is exactly the inclusion ordering.

```agda
Rel : ∀ ℓ → Allegory (lsuc ℓ) ℓ ℓ
Rel ℓ .cat .Ob      = Set ℓ
Rel ℓ .cat .Hom A B = ∣ A ∣ → ∣ B ∣ → Ω
Rel ℓ ._≤_ R S      = ∀ x y → ∣ R x y ∣ → ∣ S x y ∣
```

Relational composition is again given by the familiar formula: The
composite of $R$ and $S$ is given by the relation which "bridges the
gap", i.e. $(R \circ S)(x, z)$ iff. there exists some $y$ such that
$R(x, y)$ and $S(y, z)$. I'm not sure how surprising this will be to
some of you --- embarassingly, it was fairly surprising to me --- but
the identity relation is.. the identity relation:

```agda
Rel ℓ .cat ._∘_ S R x z = elΩ (Σ _ λ y → ∣ R x y ∣ × ∣ S y z ∣)
Rel ℓ .cat .id x y      = elΩ (x ≡ y)
```

We can investigate the reason for this by working through e.g. the proof
that relational composition is right-unital. We'll leave the identity
relation written as just $\Id$, but in either case, what we want to
show that $$(x, y) \mapsto \exists (a : A), \Id(x, a) \land R(a,
y)$$ relates the same pairs as $R$. In the interesting direction, we're
given some $a : A$ and a witness that $R(a, y)$: but what we wanted was
to show $R(x, y)$! Fortunately if we we set $\Id(x, a)$, then $R(x,
y) \simeq R(a, y)$, and we're done.

```agda
Rel ℓ .cat .idr {A} {B} R = ext λ x y → biimp
  (rec! (λ a b w → subst (λ e → ∣ R e y ∣) (sym b) w))
  (λ w → inc (x , inc refl , w))
```

The other interesting bits of the construction are meets, the dual, and
the implication witnessing the modular law: Meets are given by the
pointwise conjunction, i.e., $R \cap S$ is

- The relation given by the intersection of the subsets representing $R$
and $S$, or
- The relation which relates $x$ and $y$ iff. both $R$ and $S$ relate
them.

The dual is given by inverting the order of arguments to $R$, and the
modular law is given by some pair-shuffling.

```agda
Rel ℓ ._∩_ R S x y = el (∣ R x y ∣ × ∣ S x y ∣) (hlevel 1)
Rel ℓ ._† R x y = R y x
Rel ℓ .modular R S T x y (α , β) = case α of λ where
  z x~z z~y → inc (z , (x~z , inc (y , β , z~y)) , z~y)
```

<details>
<summary>The rest of the construction is either automated, or very
boring applications of propositional extensionality. Most of the proof
below was not written by me, but rather by Agda's^[rather precarious]
automatic proof search: that speaks to how contentful it is.</summary>

```agda
Rel ℓ .cat .Hom-set x y = hlevel 2
Rel ℓ .cat .idl R = ext λ x y → biimp
  (rec! λ z x~z z=y → subst (λ e → ∣ R x e ∣) z=y x~z)
  (λ w → inc (y , w , inc refl))

Rel ℓ .cat .assoc T S R = ext λ x y → biimp
  (rec! λ a b r s t → inc (b , r , inc (a , s , t)))
  (rec! λ a r b s t → inc (b , inc (a , r , s) , t))

Rel ℓ .≤-thin = hlevel 1
Rel ℓ .≤-refl x y w = w
Rel ℓ .≤-trans x y p q z = y p q (x p q z)
Rel ℓ .≤-antisym p q = ext λ x y → biimp (p x y) (q x y)

Rel ℓ ._◆_ f g a b = □-map (λ { (x , y , w) → x , g a x y , f x b w })

-- This is nice:
Rel ℓ .dual R = refl
Rel ℓ .dual-∘ = ext λ x y → biimp
  (□-map λ { (a , b , c) → a , c , b })
  (□-map λ { (a , b , c) → a , c , b })
Rel ℓ .dual-≤ f≤g x y w = f≤g y x w

Rel ℓ .∩-le-l x y (a , _) = a
Rel ℓ .∩-le-r x y (_ , b) = b
Rel ℓ .∩-univ h≤f h≤g x y h = h≤f x y h , h≤g x y h
```

</details>
