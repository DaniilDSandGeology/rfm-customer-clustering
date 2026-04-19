import pandas as pd
import numpy as np
from typing import Optional


def get_quantity_canceled(data):
    """Функция для создания признака количества отменённого товара (улучшенная версия).

    Использует накопительный баланс покупок/возвратов для каждого клиента и товара.

    Args:
        data (pd.DataFrame): таблица с транзакциями (должна содержать 
            CustomerID, StockCode, InvoiceDate, Quantity)

    Returns:
        pd.Series: столбец QuantityCanceled той же длины и индекса, что и data.
                   Для каждой покупки указывает, сколько из неё было отменено.
    """
    # Сортируем данные по клиенту, товару и дате
    data_sorted = data.sort_values(
        ['CustomerID', 'StockCode', 'InvoiceDate']).copy()

    # Инициализируем результат
    quantity_canceled = pd.Series(0.0, index=data_sorted.index, dtype=float)

    # Группируем по клиенту и товару
    for (customer, stock), group in data_sorted.groupby(['CustomerID', 'StockCode']):
        # Накопленный баланс (положительный = есть что отменять)
        balance = 0
        # Стек покупок для LIFO: храним (индекс, количество)
        purchase_stack = []

        for idx, row in group.iterrows():
            if row['Quantity'] > 0:  # Покупка
                purchase_stack.append((idx, row['Quantity']))
                balance += row['Quantity']
            else:  # Возврат
                cancel_qty = -row['Quantity']
                remaining_to_cancel = cancel_qty

                # Отменяем с последних покупок (LIFO)
                while remaining_to_cancel > 0 and purchase_stack:
                    purchase_idx, purchase_qty = purchase_stack.pop()

                    # Сколько можем отменить с этой покупки
                    cancel_from_this = min(remaining_to_cancel, purchase_qty)
                    quantity_canceled.loc[purchase_idx] += cancel_from_this
                    remaining_to_cancel -= cancel_from_this

                    # Если покупка не полностью отменена, возвращаем остаток в стек
                    if cancel_from_this < purchase_qty:
                        purchase_stack.append(
                            (purchase_idx, purchase_qty - cancel_from_this))

                if remaining_to_cancel > 0:
                    print(f"Warning: Для возврата {idx} осталось {remaining_to_cancel} нераспределённого товара "
                          f"(CustomerID={customer}, StockCode={stock})")
                    quantity_canceled.loc[idx] = np.nan
                else:
                    balance -= cancel_qty

    # Возвращаем в исходном порядке
    return quantity_canceled.reindex(data.index)


__all__ = ['get_quantity_canceled']
