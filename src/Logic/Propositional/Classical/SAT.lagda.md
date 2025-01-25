<!--
```agda
open import 1Lab.Prelude

open import Data.Id.Base
open import Data.Bool
open import Data.List hiding (_++_)
open import Data.Dec
open import Data.Fin using (Fin; fzero; fsuc; avoid; _[_≔_]; delete; zero; suc; fin-view)
open import Data.Nat
open import Data.Sum

open import Logic.Propositional.Classical.CNF
open import Logic.Propositional.Classical

open import Meta.Brackets

import Data.Fin as Fin
```
-->

```agda
module Logic.Propositional.Classical.SAT where
```

<!--
```agda
private variable
  Γ Δ Θ : Nat
  ψ θ ζ : Ctx Γ
  P Q R : Proposition Γ
```
-->

# SAT solving {defines="SAT-solving SAT satisfiable"}

SAT solving is the process of determining if we can find some assignment
of variables $\rho$ that makes a given formula $\phi$ in classical
propositional logic true. Such an assignment is called a **satisfying
assignment**, hence the name SAT. This is a classic problem in the field
of computer science, and many other important and interesting problems
can be reduced to finding satisfying assignments to huge formulae.

Unfortunately, SAT solving is provably hard, from a complexity
standpoint. However, this will not stop us from writing a solver
anyways! For the sake of efficiency, our solver will operate on
expressions in [[conjunctive normal form]].

# Unit propagation

The algorithm we will use is a simplified version of the classic
[DPLL]{.abbrev title="Davis-Putnam-Logemann-Loveland"} algorithm, which
combines backtracking search with a mechanism for pruning the search
space, known as **unit propagation**.

The idea behind this is the observation `unit-clause-sat`{.Agda}.
Translated into symbolic notation, it says that if our overall formula
$\phi$ breaks down as $\phi_0 \land P \land \phi_1$ --- that is, we have
a clause which consists of a single literal, then it's necessary for
$\vDash \phi$ that $\vDash P$. Not only does this give us one datum of
our satisfying assignment, it also lets us get rid of any clauses that
mention $P$, since they must also be true!

Even better, we can also remove any occurrences of $\neg P$ from our
clauses --- since we've decided that $P$ is true, having $\rm{false}$ as
a disjunct has no effect on the remaining clauses. This reduces the size
of the search space considerably, and makes the problem a bit more
tractable.

Luckily, unit propagation is rather easy to implement. The workhorse is
the recursive function `delete-literal`{.Agda}. Pay attention to its
type: expressed non-numerically, it says that given the index of $\phi$
and a formula in $\Gamma_0, \phi, \Gamma_1$, we'll return a formula in
$\Gamma_0, \Gamma_1.$

```agda
delete-literal
  : (x : Fin (suc Γ)) (ϕ : Clause (suc Γ))
  → Clause Γ
delete-literal {Γ = zero}  i ϕ  = []
delete-literal {Γ = suc Γ} i [] = []
delete-literal {Γ = suc Γ} i (x ∷ ϕ) with i ≡? lit-var x
... | yes _  = delete-literal i ϕ
... | no i≠x = avoid-lit i x i≠x ∷ delete-literal i ϕ
```

Unit propagation is then applying this function to all the clauses in a
CNF expression.

```agda
unit-propagate : Literal (suc Γ) → CNF (suc Γ) → CNF Γ
unit-propagate x [] = []
unit-propagate x (ϕ ∷ ϕs) with elem? x ϕ
... | yes _ = unit-propagate x ϕs
... | no _  = delete-literal (lit-var x) ϕ ∷ unit-propagate x ϕs
```

However, while this procedure is easy to implement, it's actually
slightly tricky to prove correct. We'll start by showing a couple of
quick lemmas regarding assignment of variables.

```agda
avoid-lit-insert
  : (x y : Literal (suc Γ)) (x≠y : ¬ (lit-var x ≡ lit-var y)) (ρ : Fin Γ → Bool)
  → ⟦ y ⟧ (ρ [ lit-var x ≔ lit-val x ])
  ≡ ⟦ avoid-lit (lit-var x) y x≠y ⟧ ρ

lit-assign-neg-false
  : (x : Literal (suc Γ)) (ρ : Fin Γ → Bool)
  → ⟦ x ⟧ (ρ [ lit-var (¬lit x) ≔ lit-val (¬lit x) ]) ≡ false

lit-assign-true
  : (x : Literal (suc Γ)) (ρ : Fin Γ → Bool)
  → ⟦ x ⟧ (ρ [ lit-var x ≔ lit-val x ]) ≡ true
```

<!--
```agda
avoid-lit-insert {Γ = Γ} x (lit y) x≠y ρ =
  Fin.avoid-insert ρ (lit-var x) (lit-val x) y x≠y
avoid-lit-insert {Γ = Γ} x (neg y) x≠y ρ =
  ap not (Fin.avoid-insert ρ (lit-var x) (lit-val x) y x≠y)

lit-assign-neg-false (lit x) ρ = Fin.insert-lookup ρ x false
lit-assign-neg-false (neg x) ρ = ap not (Fin.insert-lookup ρ x true)

lit-assign-true (lit x) ρ = Fin.insert-lookup ρ x true
lit-assign-true (neg x) ρ = ap not (Fin.insert-lookup ρ x false)
```
-->

Next, we show that deleting literals preserves the truth value of a
given assignment, as long as the literal doesn't show up in the clause.
This is not hard to show, just tedious.

```agda
delete-literal-sound
  : (x : Literal (suc Γ)) (ϕ : Clause (suc Γ))
  → ¬ (x ∈ ϕ)
  → (ρ : Fin Γ → Bool)
  → ⟦ ϕ ⟧ (ρ [ lit-var x ≔ lit-val x ]) ≡ ⟦ delete-literal (lit-var x) ϕ ⟧ ρ

delete-literal-sound {zero} x [] x∉ϕ ρ = refl
delete-literal-sound {zero} (lit i) (lit j ∷ ϕ) x∉ϕ ρ with fin-view i | fin-view j
... | zero | zero = absurd (x∉ϕ (here reflᵢ))
delete-literal-sound {zero} (lit i) (neg j ∷ ϕ) x∉ϕ ρ with fin-view i | fin-view j
... | zero | zero = delete-literal-sound (lit fzero) ϕ (x∉ϕ ∘ there) ρ
delete-literal-sound {zero} (neg i) (lit j ∷ ϕ) x∉ϕ ρ with fin-view i | fin-view j
... | zero | zero = delete-literal-sound (neg fzero) ϕ (x∉ϕ ∘ there) ρ
delete-literal-sound {zero} (neg i) (neg j ∷ ϕ) x∉ϕ ρ with fin-view i | fin-view j
... | zero | zero = absurd (x∉ϕ (here reflᵢ))

delete-literal-sound {suc Γ} x []      x∉ϕ ρ = refl
delete-literal-sound {suc Γ} x (y ∷ ϕ) x∉ϕ ρ with lit-var x ≡? lit-var y
... | yes x=y =
  ap₂ or
    (subst (λ e → ⟦ y ⟧ (ρ [ lit-var e ≔ lit-val e ]) ≡ false)
      (sym (literal-eq-negate x y (x∉ϕ ∘ _∈ₗ_.here ∘ Id≃path.from) x=y))
      (lit-assign-neg-false y ρ)) refl
  ∙ delete-literal-sound x ϕ (x∉ϕ ∘ there) ρ
... | no x≠y = ap₂ or
  (avoid-lit-insert x y x≠y ρ)
  (delete-literal-sound x ϕ (x∉ϕ ∘ there) ρ)
```

Soundness and completeness of unit propagation follow quickly.

```agda
unit-propagate-sound
  : (x : Literal (suc Γ)) (ϕs : CNF (suc Γ)) (ρ : Fin Γ → Bool)
  → ⟦ ϕs ⟧ (ρ [ lit-var x ≔ lit-val x ]) ≡ ⟦ unit-propagate x ϕs ⟧ ρ
unit-propagate-sound x []       ρ = refl
unit-propagate-sound x (ϕ ∷ ϕs) ρ with elem? x ϕ
... | yes x∈ϕ = ap₂ and
  (any-one-of (λ l → ⟦ l ⟧ (ρ [ lit-var x ≔ lit-val x ]))
    x ϕ x∈ϕ (lit-assign-true x ρ))
  (unit-propagate-sound x ϕs ρ)
... | no ¬x∉ϕ = ap₂ and
  (delete-literal-sound x ϕ ¬x∉ϕ ρ)
  (unit-propagate-sound x ϕs ρ)

unit-propagate-complete
  : (x : Literal (suc Γ)) (ϕs : CNF (suc Γ)) (ρ : Fin (suc Γ) → Bool)
  → ⟦ x  ⟧ ρ ≡ true
  → ⟦ ϕs ⟧ ρ ≡ ⟦ unit-propagate x ϕs ⟧ (delete ρ (lit-var x))
unit-propagate-complete x ϕs ρ x-true =
  ap ⟦ ϕs ⟧ (sym $ funext $
    Fin.insert-delete ρ (lit-var x) (lit-val x) (literal-sat-val x ρ x-true))
  ∙ unit-propagate-sound x ϕs (delete ρ (lit-var x))
```

Since the syntax of CNF is a very "discrete" object (it consists of
lists of lists of numbers), it's possible to write a decision procedure
which traverses a list of clauses and picks a clause consisting of a
literal, if one exists.

```agda
has-unit-clause? : (ϕs : CNF Γ) → Dec (Σ[ x ∈ Literal Γ ] ((x ∷ []) ∈ ϕs))
has-unit-clause? [] = no λ ()

has-unit-clause? ([] ∷ ϕs) with has-unit-clause? ϕs
... | yes (x , x∈ϕs) = yes (x , there x∈ϕs)
... | no  ¬ϕ-unit    = no λ where
  (x , here  ())
  (x , there x∈ϕs) → ¬ϕ-unit (x , x∈ϕs)

has-unit-clause? ((x ∷ []) ∷ ϕs)    = yes (x , here reflᵢ)
has-unit-clause? ((x ∷ y ∷ ϕ) ∷ ϕs) with has-unit-clause? ϕs
... | yes (x , x∈ϕs) = yes (x , there x∈ϕs)
... | no  ¬ϕ-unit    = no λ where
  (x , here  ())
  (x , there x∈ϕs) → ¬ϕ-unit (x , x∈ϕs)
```

We can now piece together the result which justifies unit propagation:
If we have a satisfying assignment $\rho \vDash \phi$, and a literal $x$
appears as a unit clause in $\phi$, then we know that $\rho(x)$ must be
true.

```agda
unit-clause-sat
  : (x : Literal Γ) (ϕs : CNF Γ)
  → (x ∷ []) ∈ ϕs
  → (ρ : Fin Γ → Bool)
  → ⟦ ϕs ⟧ ρ ≡ true → ⟦ x ⟧ ρ ≡ true
unit-clause-sat x (ϕ ∷ ϕs) (here [x]=ϕ) ρ ϕs-sat =
  ⟦ x ⟧ ρ      ≡⟨ sym (or-falser _) ⟩
  ⟦ x ∷ [] ⟧ ρ ≡⟨ ap (λ e → (⟦ e ⟧ ρ)) (Id≃path.to [x]=ϕ) ⟩
  ⟦ ϕ ⟧ ρ      ≡⟨ and-reflect-true-l ϕs-sat ⟩
  true         ∎
unit-clause-sat x (y ∷ ϕs) (there [x]∈ϕs) ρ ϕs-sat =
  unit-clause-sat x ϕs [x]∈ϕs ρ (and-reflect-true-r ϕs-sat)
```

We also note that it is impossible to find a satisfying assignment to a
clause with no atoms.

```agda
¬empty-clause-sat : (ϕ : Clause 0) (ρ : Fin 0 → Bool) → ⟦ ϕ ⟧ ρ ≡ true → ⊥
¬empty-clause-sat []           ρ sat = true≠false (sym sat)
¬empty-clause-sat (lit () ∷ ϕ) ρ sat
¬empty-clause-sat (neg () ∷ ϕ) ρ sat
```

Next, we observe that if the result of unit propagation is satisfiable,
then the original expression must be satisfiable. Likewise, if
the result of unit propagation is unsatisfiable, then the original
expression is unsatisfiable.

```agda
unit-propagate-sat
  : (x : Literal (suc Γ)) (ϕs : CNF (suc Γ))
  → Σ[ ρ ∈ (Fin Γ → Bool) ]       (⟦ unit-propagate x ϕs ⟧ ρ ≡ true)
  → Σ[ ρ ∈ (Fin (suc Γ) → Bool) ] (⟦ ϕs ⟧ ρ ≡ true)
unit-propagate-sat x ϕs (ρ , ρ-sat) =
    ρ [ lit-var x ≔ lit-val x ]
  , unit-propagate-sound x ϕs ρ ∙ ρ-sat

unit-propagate-unsat
  : (x : Literal (suc Γ))
  → (ϕs : CNF (suc Γ))
  → ¬ (Σ[ ρ ∈ (Fin Γ → Bool) ]       ⟦ unit-propagate x ϕs ⟧ ρ ≡ true)
  → ¬ (Σ[ ρ ∈ (Fin (suc Γ) → Bool) ] ⟦ ϕs ⟧ ρ ≡ true × ⟦ x ⟧ ρ ≡ true)
unit-propagate-unsat x ϕs ¬sat (ρ , ρ-sat , x-sat) = ¬sat $
    delete ρ (lit-var x)
  , sym (unit-propagate-complete x ϕs ρ x-sat) ∙ ρ-sat
```

Armed with these lemmas, we can finally write our SAT solver.
We shall perform induction on the number of atoms in our CNF
expression, $\phi$. If $\phi$ has no atoms, then we know its
satisfiability by looking at the clauses: having no clauses is the CNF
representation of $\top$, while having an empty clause is the CNF
representation of $\bot$.

```agda
cnf-sat? : (ϕs : CNF Γ) → Dec (Σ[ ρ ∈ (Fin Γ → Bool) ] ⟦ ϕs ⟧ ρ ≡ true)
cnf-sat? {Γ = zero} []       = yes ((λ ()) , refl)
cnf-sat? {Γ = zero} (ϕ ∷ ϕs) = no λ where
  (ρ , sat) → ¬empty-clause-sat ϕ ρ (and-reflect-true-l sat)
```

The interesting case is when $\phi$ contains at least one atom. Ideally,
$\phi$ has a unit clause, in which case we can perform unit propagation,
and recursively and check satisfiability of the (hopefully much
smaller!) resulting formula.

```agda
cnf-sat? {Γ = suc Γ} ϕs with has-unit-clause? ϕs
... | yes (x , [x]∈ϕs) with cnf-sat? (unit-propagate x ϕs)
...   | yes sat = yes (unit-propagate-sat x ϕs sat)
...   | no ¬sat = no λ (ρ , ρ-sat) →
        unit-propagate-unsat x ϕs ¬sat
          (ρ , ρ-sat , unit-clause-sat x ϕs [x]∈ϕs ρ ρ-sat)
```

If there aren't, then we're slightly out of luck. We'll have to guess:
We pick the first atom in $\phi$ and arbitrarily decide that it must be
true. We can then propagate this choice, using the same unit-propagation
procedure, to obtain a formula $\phi[\rm{true}/0]$. _If_ this formula is
satisfiable, we know that $\phi$ is, too.

```agda
cnf-sat? {Γ = suc Γ} ϕs
    | no ¬ϕs-unit with cnf-sat? (unit-propagate (lit fzero) ϕs)
...   | yes sat-true = yes (unit-propagate-sat (lit fzero) ϕs sat-true)
```

But choosing $\rm{true}$ was a guess, and it might have been wrong! If
it was, then we'll try again: we'll check satisfiability of
$\phi[\rm{false}/0]$. Once again, satisfiability of this new formula
implies satisfiability of the original.

```agda
...   | no ¬sat-true with cnf-sat? (unit-propagate (neg fzero) ϕs)
...     | yes sat-false = yes (unit-propagate-sat (neg fzero) ϕs sat-false)
```

However, we might have been wrong again! It might be the case that the
expression is *un*satisfiable. In this case we can return "UNSAT": if
anyone claims to have a satisfying assignment $\rho \vDash \phi$, they
must have chosen one of two values for $\rho(0)$, and we know that
neither can work.

```agda
...     | no ¬sat-false = no λ (ρ , ρ-sat) → Bool-elim (λ b → ρ fzero ≡ b → ⊥)
  (λ ρ₀-true  → unit-propagate-unsat (lit fzero) ϕs ¬sat-true
    (ρ , ρ-sat , ρ₀-true))
  (λ ρ₀-false → unit-propagate-unsat (neg fzero) ϕs ¬sat-false
    (ρ , ρ-sat , ap not ρ₀-false))
  (ρ fzero) refl
```

And that's it! Note that "classic" DPLL also includes a second rule
known as _pure literal elimination_. The idea here is that if a literal
only shows up as negated or not negated, then we can delete all
occurrences of that literal. However, this operation is somewhat
expensive to perform, and also rather annoying to program in Agda.
Therefore, it has been omitted from this implementation.
