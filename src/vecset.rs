
/// `VecSet` is a set backed by a vector, intended to be used for small sets.
struct VecSet<T: Eq> {
    pub data: Vec<T>,
}

impl<T> VecSet<T> 
where T: Eq {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity) }
    }

    pub fn insert(&mut self, value: T) {
        if !self.contains(&value) {
            self.data.push(value);
        }
    }

    pub fn contains(&self, value: &T) -> bool {
        self.data.contains(value)
    }

    pub fn remove(&mut self, value: &T) {
        let target = self.data.iter().position(|x| x == value);
        if let Some(target) = target {
            self.data[target] = self.data.pop().unwrap();
        }
    }

    pub unsafe fn insert_unchecked(&mut self, value: T) {
        self.data.push(value);
    }

    pub unsafe fn remove_unchecked(&mut self, value: &T) {
        let target = self.data
            .iter()
            .position(|x| x == value)
            .unwrap_unchecked();
        *self.data.get_unchecked_mut(target) = self.data.pop().unwrap_unchecked();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }
}