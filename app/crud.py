from sqlalchemy.orm import Session
from app.models_db import Generation


def save_generation(
    db: Session,
    prompt: str,
    generated_text: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    response_time_ms: float,
) -> Generation:
    record = Generation(
        prompt=prompt,
        generated_text=generated_text,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        response_time_ms=response_time_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_all_generations(db: Session, limit: int = 50) -> list[Generation]:
    return (
        db.query(Generation)
        .order_by(Generation.created_at.desc())
        .limit(limit)
        .all()
    )


def get_generation_by_id(db: Session, generation_id: int) -> Generation | None:
    return db.query(Generation).filter(Generation.id == generation_id).first()


def delete_generation(db: Session, generation_id: int) -> bool:
    record = get_generation_by_id(db, generation_id)
    if not record:
        return False
    db.delete(record)
    db.commit()
    return True