
class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            print("*** Dequeue ",self.items[0].commit_id)
            return self.items.pop(0) #element at the front of queue
        else:
            raise IndexError("Queue is empty")

    def get_first(self):
        if not self.is_empty():
            print("$$$ First ",self.items[0].commit_id)

    def size(self):
        return len(self.items)

    def contains(self, item):
        for index, element in enumerate(self.items):
            if element == item:
                return index, element  # Return both the index and the matched element
        return None, None  # Return None if the item is not found

    def remove(self, item):
        index, _ = self.contains(item)
        if index is not None:
            del self.items[index]

    def __iter__(self):
        return iter(self.items)

    def print_queue(self):
        print("Elements in Queue!")
        for index, element in enumerate(self.items):
            print(index,element.commit_id)
        print("Queue finished!")


def test_queue():
    my_queue = Queue()
    print(my_queue.is_empty())  # True
    # Enqueue some elements
    my_queue.enqueue(1)
    my_queue.enqueue(2)
    my_queue.enqueue(3)
    my_queue.check_queued_elements()
    print("Size",my_queue.size())  # 3
    print("Element",my_queue.dequeue())  # 1
    print("Element",my_queue.dequeue())  # 2
    print("Size",my_queue.size())   # 1
    # Dequeue the last element
    print("Element",my_queue.dequeue())  # 3
    print(my_queue.is_empty())  # True

